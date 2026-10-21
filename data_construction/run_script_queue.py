"""Public construction draft: run_script_queue.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/run_script_queue.py [options]\n\n  --film\n  --films  清单文件，每行一个 slug；只跑这些\n  --redo  已完成的也重跑\n  -j, --workers\n  --prep-only  只生成输入，不跑 agent\n  --all  Process all inputs explicitly\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (
        CODE_ROOT, DATA_ROOT, require_target, validate_data_slug, export_slug, select_films, annotation_process
    )
    from .annotation_backend import build_annotation_command, AnnotationEvents
    from .annotation_visuals import prepare_script_visuals
    from .annotation_script_stages import (
        build_shot_prompt, validate_draft, build_structure_prompt,
        merge_structure, atomic_write_json, validate_prefix, make_shot_batch,
        canonicalize_batch_lines, SHOT_BATCH_SIZE,
    )
except ImportError:  # Direct script invocation.
    from release_config import (
        CODE_ROOT, DATA_ROOT, require_target, validate_data_slug, export_slug, select_films, annotation_process
    )
    from annotation_backend import build_annotation_command, AnnotationEvents
    from annotation_visuals import prepare_script_visuals
    from annotation_script_stages import (
        build_shot_prompt, validate_draft, build_structure_prompt,
        merge_structure, atomic_write_json, validate_prefix, make_shot_batch,
        canonicalize_batch_lines, SHOT_BATCH_SIZE,
    )

import argparse, glob, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = DATA_ROOT
EFFORT = "max"   # 地基产物，后面每一步都以它为准
TOOLS = ("Read,Write,Edit,Glob,Grep,"
         "Bash(ffmpeg:*),Bash(ffprobe:*),Bash(python3:*),Bash(ls:*),Bash(cat:*),Bash(mkdir:*)")
STALL_S = 20 * 60
TRIES = 2
MAX_WAITS = 3


def _draft_count(path, slug):
    """Best-effort progress only; final coverage and dialogue checks remain separate."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("film") != slug or not isinstance(document.get("shots"), list):
            return None
        return len({row["id"] for row in document["shots"]
                    if isinstance(row, dict) and isinstance(row.get("id"), str)
                    and isinstance(row.get("desc"), str) and row["desc"].strip()})
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def prep(slug):
    """assemble the measured facts the agent is not allowed to re-derive"""
    validate_data_slug(ROOT, slug)
    E, T = f"{ROOT}/exports/{slug}", f"{ROOT}/edit/task_{slug}"
    os.makedirs(f"{T}/stages", exist_ok=True)
    os.makedirs(f"{T}/reference", exist_ok=True)
    os.makedirs(f"{T}/work", exist_ok=True)
    pb = json.load(open(f"{E}/probe.json"))
    ref = f"{T}/reference/golden.mp4"
    src = os.path.abspath(f"{pb['dir']}/{pb['golden']}")
    if os.path.lexists(ref) and os.path.realpath(ref) != src:
        os.remove(ref)
    if not os.path.lexists(ref):
        os.symlink(src, ref)

    units = {u["id"]: u for u in json.load(open(f"{T}/s3_units.json"))["units"]}
    import csv
    shots = []
    for r in csv.DictReader(open(f"{E}/golden_clips/manifest.csv")):
        u = units.get(r["id"], {})
        shots.append({"id": r["id"], "start_s": round(float(r["start_s"]), 3),
                      "dur_s": round(float(r["dur_s"]), 3),
                      "kind": u.get("kind", "source"),
                      "card_stills": sorted(os.path.basename(x) for x in
                                            glob.glob(f"{T}/cards/{r['id']}_s*.png"))
                      if u.get("kind") == "card" else None})

    tr = json.load(open(f"{E}/golden_transcript.json"))
    speech = [{"start_s": round(float(s["start"]), 2), "end_s": round(float(s["end"]), 2),
               "text": s["text"].strip()} for s in tr["segments"] if s["text"].strip()]

    W = sorted((float(w["s"]), float(w["e"]), w["w"])
               for sg in tr["segments"] for w in sg["words"] if w.get("p", 1) > 0.4)
    for sh in shots:
        a, b = sh["start_s"], sh["start_s"] + sh["dur_s"]
        inside = [w for w in W if w[0] < b - 0.02 and w[1] > a + 0.02]
        if not inside:
            sh["line"] = None; continue
        if sh["kind"] == "card":
            # A card carries no dialogue. What lands here is the tail of the previous
            # line spilling over — real, but it belongs to the cut, not the card, and
            # in the shot log it reads as if the card had a spoken line.
            sh["line"] = None; continue
        sh["line"] = {
            "text": " ".join(w[2] for w in inside),
            # a line that starts before this shot, or runs past its end, is shared with
            # the neighbour — that is what makes a cut a J or an L
            "from_prev": round(max(0.0, a - inside[0][0]), 2) or None,
            "into_next": round(max(0.0, inside[-1][1] - b), 2) or None,
        }

    cl = {c["idx"]: c for c in json.load(open(f"{T}/stages/cut_list.json"))["cuts"]}
    music = []
    for c in json.load(open(f"{E}/audio_cuts.json"))["cuts"]:
        tags = c.get("music") or []
        if not tags:
            continue
        k = cl.get(c["idx"], {})
        music.append({"at_s": round(k.get("timeline_s", c.get("t", 0)), 2),
                      "after_shot": k.get("after"), "before_shot": k.get("before"),
                      "tags": tags})

    out = {"film": slug, "n_shots": len(shots),
           "note": "这些是量出来的，不要重新推导；画面内容要你自己看",
           "shots": shots, "speech": speech, "music": music}
    json.dump(out, open(f"{T}/stages/script_inputs.json", "w"),
              ensure_ascii=False, indent=1)
    return T, len(shots)


def check(slug, T):
    """a shot log that skips shots, or beats that leave gaps, is not a shot log"""
    p = f"{T}/stages/script.json"
    if not os.path.exists(p):
        return ["script.json 未生成"]
    try:
        with open(p, encoding="utf-8") as stream:
            d = json.load(stream)
    except Exception as e:
        return [f"JSON 解析失败: {e}"]
    with open(f"{T}/stages/script_inputs.json", encoding="utf-8") as stream:
        inp = json.load(stream)
    return check_document(slug, inp, d)


def check_document(slug, inp, d):
    """Validate a complete candidate in memory before publishing script.json."""
    if not isinstance(d, dict):
        return ["Script must be a JSON object"]
    if not isinstance(d.get("shots"), list) or any(not isinstance(s, dict) for s in d["shots"]):
        return ["Script shots must be an array of objects"]
    for key in ("beats", "chunks"):
        if not isinstance(d.get(key), list) or any(not isinstance(row, dict) for row in d[key]):
            return [f"Script {key} must be an array of objects"]
    want = [s["id"] for s in inp["shots"]]
    kind = {s["id"]: s["kind"] for s in inp["shots"]}
    lines = {s["id"]: s["line"].get("text") if isinstance(s.get("line"), dict)
             else s.get("line") for s in inp["shots"]}
    bad = []
    if d.get("film") != slug or d.get("film") != inp.get("film"):
        bad.append("film 必须与输入及请求的 slug 一致")

    got = [s.get("id") for s in d.get("shots", [])]
    if got != want:
        miss = [i for i in want if i not in got]
        extra = [i for i in got if i not in want]
        bad.append(f"镜头表对不上: {len(got)}/{len(want)} 条"
                   + (f" 缺{miss[:4]}" if miss else "") + (f" 多{extra[:4]}" if extra else "")
                   + ("" if miss or extra else " 顺序不一致"))
    for s in d.get("shots", []):
        if s.get("id") in lines and ("line" not in s or s["line"] != lines[s["id"]]):
            bad.append(f"{s.get('id')} 的 line 必须原样保留输入对白 text 或 null"); break
        if not (s.get("desc") or "").strip():
            bad.append(f"{s.get('id')} 缺画面描述"); break
        if kind.get(s.get("id")) == "card" and not (s.get("card_text") or "").strip() \
                and not (s.get("card_misflag") or "").strip():
            bad.append(f"{s.get('id')} 是字幕卡但没抄字（真不是卡就写 card_misflag）"); break
    if not d.get("logline"):
        bad.append("缺 logline")

    beats = d.get("beats") or []
    if len(beats) < 2:
        bad.append(f"只划了 {len(beats)} 段")
    idx = {s: i for i, s in enumerate(want)}
    cover, prev_end = [], -1
    for b in beats:
        sh = b.get("shots") or []
        if len(sh) != 2 or sh[0] not in idx or sh[1] not in idx:
            bad.append(f"{b.get('id')} 的 shots 不是两个合法镜头 id"); break
        a, z = idx[sh[0]], idx[sh[1]]
        if z < a:
            bad.append(f"{b.get('id')} 起止颠倒"); break
        if a != prev_end + 1:
            bad.append(f"{b.get('id')} 起点 {sh[0]} 与上一段"
                       + ("有空隙" if a > prev_end + 1 else "重叠")); break
        prev_end = z
        cover.append((a, z))
        for k in ("name", "what", "function", "why_here"):
            if not (b.get(k) or "").strip():
                bad.append(f"{b.get('id')} 缺 {k}"); break
    if not bad and prev_end != len(want) - 1:
        bad.append(f"段落没覆盖到最后一个镜头（停在第 {prev_end+1} 个）")
    if bad:
        return bad

    # Subdivide narrative beats into manageable editing chunks without crossing beat boundaries.

    beat_of = {}
    for b in beats:
        a, z = idx[b["shots"][0]], idx[b["shots"][1]]
        for i in range(a, z + 1):
            beat_of[i] = b.get("id")
    chunks = d.get("chunks") or []
    if not chunks:
        return ["缺 chunks（作业段）"]
    prev_end = -1
    for ch in chunks:
        sh = ch.get("shots") or []
        if len(sh) != 2 or sh[0] not in idx or sh[1] not in idx:
            bad.append(f"{ch.get('id')} 的 shots 不是两个合法镜头 id"); break
        a, z = idx[sh[0]], idx[sh[1]]
        if z < a:
            bad.append(f"{ch.get('id')} 起止颠倒"); break
        if a != prev_end + 1:
            bad.append(f"{ch.get('id')} 起点 {sh[0]} 与上一段"
                       + ("有空隙" if a > prev_end + 1 else "重叠")); break
        if beat_of[a] != beat_of[z]:
            bad.append(f"{ch.get('id')} 跨越了段落边界（{beat_of[a]} → {beat_of[z]}）"); break
        if ch.get("beat") != beat_of[a]:
            bad.append(f"{ch.get('id')} 标的 beat 是 {ch.get('beat')}，实际落在 {beat_of[a]}"); break
        # Size is a target, not a rule. Forcing a split to hit exactly five puts the
        # boundary somewhere the picture does not actually turn, which is worse than a
        # eight-shot chunk cut at a real seam. Only absurd sizes are rejected; the
        # distribution is printed so drift is visible.
        n = z - a + 1
        blen = sum(1 for i in beat_of if beat_of[i] == beat_of[a])
        if n > 15 and blen > 15:
            bad.append(f"{ch.get('id')} 有 {n} 镜，这不算划分了"); break
        if not (ch.get("what") or "").strip():
            bad.append(f"{ch.get('id')} 缺 what"); break
        prev_end = z
    if not bad and prev_end != len(want) - 1:
        bad.append(f"作业段没覆盖到最后一个镜头（停在第 {prev_end+1} 个）")
    return bad


def _claude_phase(command, T, log_dir, slug, name, n, draft_path=None):
    """Run one fresh native session with its own deadline and private logs."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    events = AnnotationEvents(command.backend)
    started, ntool = time.monotonic(), 0
    stopped, killed = threading.Event(), []
    last = [time.monotonic()]
    errors, returncode, watcher = [], None, None
    try:
        with annotation_process(command.argv, cwd=T, stderr_path=log_dir / "stderr.log") as process:
            def watch_silence():
                while not stopped.wait(30):
                    if process.poll() is not None:
                        return
                    if time.monotonic() - last[0] > STALL_S:
                        killed.append(True)
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        return
            watcher = threading.Thread(target=watch_silence, daemon=True)
            watcher.start()
            with (log_dir / "session.jsonl").open("w", encoding="utf-8") as raw, \
                 (log_dir / "progress.log").open("w", encoding="utf-8") as progress:
                draft_stamp = None
                for line in process.stdout:
                    raw.write(line); raw.flush()
                    last[0] = time.monotonic()
                    for tool in events.feed(line):
                        ntool += 1
                        args = tool["input"]
                        value = args.get("file_path") or args.get("filePath") or args.get("command") or ""
                        one = " ".join(str(value).split())
                        one = "…" + one[-72:] if len(one) > 74 and "/" in one else one[:80]
                        progress.write(f"[{time.monotonic()-started:6.0f}s] → {tool['name']}  {one}\n")
                        progress.flush()
                    if draft_path is not None:
                        try:
                            stat = Path(draft_path).stat()
                            stamp = (stat.st_mtime_ns, stat.st_size)
                        except OSError:
                            continue
                        if stamp != draft_stamp:
                            draft_stamp = stamp
                            count = _draft_count(draft_path, slug)
                            if count is not None:
                                message = f"draft contains {count}/{n} shot descriptions (not yet validated)"
                                progress.write(f"[{time.monotonic()-started:6.0f}s] {message}\n")
                                progress.flush()
                                print(f"  [{name}] {message}", flush=True)
            returncode = process.wait()
    except (OSError, TimeoutError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        stopped.set()
        if watcher is not None:
            watcher.join()
    errors.extend(events.failures())
    if returncode is not None and returncode != 0:
        errors.insert(0, f"Annotation process exited with status {returncode}; see {log_dir}/stderr.log")
    if killed:
        errors.insert(0, "Annotation process exceeded its inactivity limit")
    return {"errors": errors, "completed": events.completed,
            "seconds": round(time.monotonic() - started, 3), "tools": ntool,
            "cost_usd": events.cost, "rate_limit": events.rate_limit,
            "log_dir": str(log_dir)}


def _restore_bytes(path, contents):
    """Restore an exact pre-session snapshot without publishing a partial file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _claude_once(slug, name, T, n, runs, st, resume=True, why=None):
    """Validate independent shot batches, then request structure in a fresh session."""
    # Different attempts can start in the same second; never reuse structure output.
    run_dir = Path(tempfile.mkdtemp(prefix="staged-", dir=runs))
    inputs_path = Path(T) / "stages/script_inputs.json"
    final_path = Path(T) / "stages/script.json"
    input_bytes = inputs_path.read_bytes()
    inputs = json.loads(input_bytes)
    original_final = final_path.read_bytes() if final_path.exists() else None
    started = time.monotonic()
    phases, errors, visuals, published = {}, [], None, False
    model = None
    draft_path, trusted_draft_bytes = None, None

    def unchanged(path, expected):
        try:
            return Path(path).read_bytes() == expected
        except OSError:
            return False

    def guard_inputs():
        bad = []
        if not unchanged(inputs_path, input_bytes):
            bad.append("Measured script_inputs.json was changed or removed during annotation")
        current = final_path.read_bytes() if final_path.exists() else None
        if current != original_final:
            bad.append("Model wrote stages/script.json before validated Python publication")
        if trusted_draft_bytes is not None and not unchanged(draft_path, trusted_draft_bytes):
            bad.append("Previously validated global shot prefix was changed or removed")
        return bad

    def phase(which, prompt, output_path=None, count=None):
        nonlocal model
        if why:
            prompt += ("\n\n## Previous attempt was rejected\n\n"
                       "Address these validation failures without modifying measured inputs "
                       "or frozen shot annotations:\n" +
                       "\n".join(f"- {str(error)[:500]}" for error in why[:5]) + "\n")
        log_dir = run_dir / which
        log_dir.mkdir()
        (log_dir / "PROMPT.md").write_text(prompt, encoding="utf-8")
        command = build_annotation_command(prompt, T, tools=TOOLS, effort=EFFORT)
        if command.backend != "claude":
            raise ValueError("Annotation backend changed during the staged Claude run")
        model = command.model
        print(f"  [{name}] Claude {which} → {log_dir}/progress.log", flush=True)
        result = _claude_phase(command, T, log_dir, slug, name,
                               n if count is None else count, output_path)
        phases[which] = result
        errors.extend(result["errors"])
        errors.extend(guard_inputs())

    try:
        print(f"  [{name}] preparing contact sheets for {n} shots", flush=True)
        visuals = prepare_script_visuals(T)
        errors.extend(guard_inputs())
        draft_path = Path(visuals["draft_path"])
        draft = {"film": inputs["film"], "shots": []}
        if not resume and draft_path.exists():
            shutil.copyfile(draft_path, run_dir / "previous-draft.json")
            draft_path.unlink()
            print(f"  [{name}] --redo starts a new shot inspection; previous draft archived", flush=True)
        if resume and draft_path.exists():
            try:
                candidate = json.loads(draft_path.read_bytes())
                prefix_errors = validate_prefix(inputs, candidate)
                if not prefix_errors:
                    draft = candidate
                    phases["shots_resume"] = {"resumed": True, "draft": str(draft_path),
                                              "shots": len(draft["shots"])}
                    print(f"  [{name}] resuming matching prefix {len(draft['shots'])}/{n}; contract checks "
                          "do not establish visual accuracy", flush=True)
                else:
                    errors.extend([f"Cannot resume invalid shot prefix: {error}" for error in prefix_errors])
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"Cannot resume unreadable shot prefix: {exc}; use --redo to restart inspection")
        if not errors:
            working = draft_path.parent / f"attempt-{run_dir.name}"
            if not working.resolve().is_relative_to(Path(T).resolve()):
                raise ValueError("Staged annotation outputs must remain inside the task directory")
            working.mkdir()
            if not draft_path.exists():
                atomic_write_json(draft_path, draft)
            trusted_draft_bytes = draft_path.read_bytes()
            manifest = json.loads(Path(visuals["manifest_path"]).read_bytes())
            while len(draft["shots"]) < len(inputs["shots"]) and not errors:
                start = len(draft["shots"])
                # Continue a partial sheet first, then keep the original 12-shot boundaries.
                stop = min(len(inputs["shots"]), (start // SHOT_BATCH_SIZE + 1) * SHOT_BATCH_SIZE)
                batch_name = f"shots-{start // SHOT_BATCH_SIZE + 1:03d}"
                batch_dir = working / batch_name
                batch_dir.mkdir()
                batch_inputs_path = batch_dir / "inputs.json"
                batch_manifest_path = batch_dir / "manifest.json"
                batch_output_path = batch_dir / "shots.json"
                batch_inputs, batch_manifest = make_shot_batch(
                    inputs, manifest, start, stop, task_dir=T,
                    inputs_path=batch_inputs_path, output_path=batch_output_path)
                atomic_write_json(batch_inputs_path, batch_inputs)
                atomic_write_json(batch_manifest_path, batch_manifest)
                frozen_batch = {batch_inputs_path: batch_inputs_path.read_bytes(),
                                batch_manifest_path: batch_manifest_path.read_bytes()}
                batch_visuals = {"manifest_path": batch_manifest_path, "draft_path": batch_output_path}
                phase(batch_name, build_shot_prompt(batch_inputs_path, batch_visuals),
                      batch_output_path, count=stop - start)
                for path, expected in frozen_batch.items():
                    if not unchanged(path, expected):
                        errors.append(f"Frozen batch {path.name} was changed or removed in {batch_name}")
                if batch_output_path.exists():
                    shutil.copyfile(batch_output_path, run_dir / batch_name / "shots.json")
                if not errors:
                    try:
                        batch = canonicalize_batch_lines(batch_inputs, json.loads(batch_output_path.read_bytes()))
                        errors.extend(validate_draft(batch_inputs, batch))
                    except (OSError, ValueError, TypeError) as exc:
                        errors.append(f"Shot batch output missing or invalid: {exc}")
                if not errors:
                    atomic_write_json(run_dir / batch_name / "shots.canonical.json", batch)
                    draft = {"film": inputs["film"], "shots": draft["shots"] + batch["shots"]}
                    errors.extend(validate_prefix(inputs, draft))
                    if not errors:
                        atomic_write_json(draft_path, draft)
                        trusted_draft_bytes = draft_path.read_bytes()
                        print(f"  [{name}] validated shot prefix {len(draft['shots'])}/{n} "
                              "(data contract only; visual accuracy is not certified)", flush=True)
            if not errors:
                errors.extend(validate_draft(inputs, draft))
        if not errors:
            frozen_path = working / "shots.frozen.json"
            atomic_write_json(frozen_path, {"film": draft["film"], "shots": draft["shots"]})
            frozen_bytes = frozen_path.read_bytes()
            frozen = json.loads(frozen_bytes)
            atomic_write_json(run_dir / "shots.frozen.json", frozen)
            structure_path = working / "script.structure.json"
            phase("structure", build_structure_prompt(frozen_path, inputs_path, structure_path))
            if not unchanged(frozen_path, frozen_bytes):
                errors.append("Frozen shot draft was changed or removed during structure annotation")
            if structure_path.exists():
                shutil.copyfile(structure_path, run_dir / "script.structure.json")
            if not errors:
                structure = json.loads(structure_path.read_bytes())
                document = merge_structure(frozen, structure)
                errors.extend(check_document(slug, inputs, document))
                if not errors:
                    atomic_write_json(final_path, document)
                    published = True
                    atomic_write_json(run_dir / "script.json", document)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if trusted_draft_bytes is not None and not unchanged(draft_path, trusted_draft_bytes):
            if draft_path.exists():
                shutil.copyfile(draft_path, run_dir / "draft.modified.json")
            _restore_bytes(draft_path, trusted_draft_bytes)
        if not unchanged(inputs_path, input_bytes):
            if inputs_path.exists():
                shutil.copyfile(inputs_path, run_dir / "inputs.modified.json")
            _restore_bytes(inputs_path, input_bytes)
        # A CLI may disobey the output path. Never leave its unvalidated final file
        # in place, or destroy a previously published result on a failed --redo.
        if not published:
            current = final_path.read_bytes() if final_path.exists() else None
            if current != original_final:
                if original_final is None:
                    final_path.unlink(missing_ok=True)
                else:
                    _restore_bytes(final_path, original_final)

    executed = [p for p in phases.values() if not p.get("resumed")]
    reported_costs = [p["cost_usd"] for p in executed]
    cost = sum(reported_costs) if reported_costs and all(v is not None for v in reported_costs) else None
    rate_limit = next((p["rate_limit"] for p in reversed(executed) if p.get("rate_limit")), {})
    state = {"ok": not errors and published, "at": time.strftime("%F %T"),
             "min": round((time.monotonic() - started) / 60, 1), "backend": "claude",
             "model": model, "cost_usd": cost, "tools": sum(p["tools"] for p in executed),
             "errors": errors[:5], "rate_limit": rate_limit.get("status"),
             "phases": phases, "run_dir": str(run_dir)}
    if visuals:
        state.update(visual_manifest=str(visuals["manifest_path"]), draft=str(visuals["draft_path"]))
    st["script"] = state
    atomic_write_json(Path(ROOT) / "exports" / slug / "state.json", st)
    print(f"  [{name}] " + ("✓ shot and structure stages validated" if state["ok"]
                            else f"✗ {(errors or ['No validated final output'])[0]}"), flush=True)
    return state["ok"], rate_limit, errors


def _once(slug, name, T, n, attempt, why=None, resume=True):
    validate_data_slug(ROOT, slug)
    E = f"{ROOT}/exports/{slug}"
    st = json.load(open(f"{E}/state.json"))
    st["script"] = {"ok": False, "at": time.strftime("%F %T"), "errors": ["Annotation attempt incomplete"]}
    json.dump(st, open(f"{E}/state.json", "w"), ensure_ascii=False, indent=1)
    runs = f"{ROOT}/edit/runs/{slug}/{time.strftime('%Y%m%d_%H%M%S')}_script"
    os.makedirs(runs, exist_ok=True)
    out_p = f"{T}/stages/script.json"
    pre_m = os.path.getmtime(out_p) if os.path.exists(out_p) else 0.0

    prompt = open(f"{CODE_ROOT}/prompts/PROMPT_SCRIPT.md").read()
    prompt += f"\n\n## 本片\n\n《{name}》，共 **{n} 个镜头**。slug 是 `{slug}`。\n"
    if why:
        prompt += ("\n## 上一次被打回了\n\n原因：\n- " + "\n- ".join(why[:5]) +
                   "\n\n这次务必避开。\n")
    command = build_annotation_command(prompt, T, tools=TOOLS, effort=EFFORT)
    if command.backend == "claude":
        return _claude_once(slug, name, T, n, runs, st, resume=resume, why=why)
    open(f"{runs}/PROMPT.md", "w").write(prompt)

    t0 = time.time()
    print(f"  [{name}] {n} 镜头 启动{'（重试）' if attempt else ''} → {runs}/progress.log",
          flush=True)
    events = AnnotationEvents(command.backend)
    with annotation_process(command.argv, cwd=T, stderr_path=f"{runs}/stderr.log") as p:

        last = [time.time()]; killed = []

        def _watchdog():
            while p.poll() is None:
                time.sleep(30)
                if time.time() - last[0] > STALL_S:
                    killed.append(True)
                    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception: pass
                    return
        threading.Thread(target=_watchdog, daemon=True).start()

        with open(f"{runs}/session.jsonl", "w") as raw, open(f"{runs}/progress.log", "w") as prog:
            ntool = 0
            for line in p.stdout:
                raw.write(line); raw.flush()
                last[0] = time.time()
                for tool in events.feed(line):
                    ntool += 1
                    args = tool["input"]
                    value = args.get("file_path") or args.get("filePath") or args.get("command") or ""
                    one = " ".join(str(value).split())
                    one = ("…" + one[-72:]) if (len(one) > 74 and "/" in one) else one[:80]
                    prog.write(f"[{time.time()-t0:6.0f}s] → {tool['name']}  {one}\n")
                    prog.flush()
        returncode = p.wait()
    cost, rl = events.cost, [events.rate_limit]
    if killed:
        print(f"  [{name}] ⚠ 静默超过 {STALL_S//60} 分钟，已终止", flush=True)
    dt = time.time() - t0

    bad = events.failures() + check(slug, T)
    if returncode != 0:
        bad.insert(0, f"Annotation process exited with status {returncode}; see stderr.log")
    if killed:
        bad.insert(0, "Annotation process exceeded its inactivity limit")
    if not bad and (not os.path.exists(out_p) or os.path.getmtime(out_p) <= pre_m):
        bad = ["script.json 没被本次运行写过"]
    if os.path.exists(out_p):
        shutil.copy(out_p, f"{runs}/script.json")
    st["script"] = {"ok": not bad, "at": time.strftime("%F %T"), "min": round(dt / 60, 1),
                    "backend": command.backend, "model": command.model,
                    "cost_usd": cost, "tools": ntool, "errors": bad[:5],
                    "rate_limit": rl[0].get("status")}
    json.dump(st, open(f"{E}/state.json", "w"), ensure_ascii=False, indent=1)

    if bad:
        print(f"  [{name}] ✗ {bad[0][:160]}", flush=True)
        return False, rl[0], bad
    _d = json.load(open(out_p)) if os.path.exists(out_p) else {}
    nb, nc = len(_d.get("beats") or []), len(_d.get("chunks") or [])
    _ids = [x["id"] for x in json.load(open(f"{T}/stages/script_inputs.json"))["shots"]]
    _sz = sorted(_ids.index(c["shots"][1]) - _ids.index(c["shots"][0]) + 1
                 for c in (_d.get("chunks") or [])
                 if c.get("shots") and c["shots"][0] in _ids and c["shots"][1] in _ids)
    _rng = f" {_sz[0]}-{_sz[-1]}镜" if _sz else ""
    print(f"  [{name}] {dt/60:.1f}分 {ntool}次" + (f" ${cost:.2f}" if cost else "") +
          (f"  {nb} 段 / {nc} 作业段{_rng}  ✓" if not bad else f"  ✗ {bad[0][:70]}"),
          flush=True)
    return (not bad, rl[0], bad)


def run(slug, name, redo=False):
    """never raises — one film must not take the queue with it"""
    validate_data_slug(ROOT, slug)
    try:
        E = f"{ROOT}/exports/{slug}"
        st = json.load(open(f"{E}/state.json"))
        for need in ("cutlist", "acuts"):
            if not st.get(need, {}).get("ok"):
                print(f"  跳过 {name}: 缺 {need}", flush=True); return False
        if st.get("script", {}).get("ok") and not redo:
            print(f"  跳过 {name}: 场记表已完成", flush=True); return True
        T, n = prep(slug)
        attempt = waits = 0
        why = None
        while attempt < TRIES:
            ok, rl, why = _once(slug, name, T, n, attempt, why,
                                resume=not redo or attempt > 0 or waits > 0)
            if ok:
                return True
            if rl.get("status") not in (None, "allowed") and waits < MAX_WAITS:
                wake = (rl.get("resetsAt") or (time.time() + 900)) + 60
                print(f"  [{name}] 触到限流，等到 "
                      f"{time.strftime('%H:%M', time.localtime(wake))} 再试", flush=True)
                while time.time() < wake:
                    time.sleep(60)
                waits += 1
                continue
            attempt += 1
        return False
    except Exception as e:
        print(f"  [{name}] ✗ 未捕获异常: {type(e).__name__}: {str(e)[:160]}", flush=True)
        try:
            state_path = Path(ROOT) / "exports" / slug / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["script"] = {"ok": False, "at": time.strftime("%F %T"),
                               "errors": [f"{type(e).__name__}: {str(e)[:500]}"]}
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        except (OSError, ValueError) as state_error:
            print(f"  [{name}] cannot record failure state: {state_error}", file=sys.stderr, flush=True)
        return False


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--film")
    P.add_argument("--films", help="清单文件，每行一个 slug；只跑这些")
    P.add_argument("--redo", action="store_true", help="已完成的也重跑")
    P.add_argument("-j", "--workers", type=int, default=1)
    P.add_argument("--prep-only", action="store_true", help="只生成输入，不跑 agent")
    P.add_argument("--all", action="store_true", help="Process all inputs explicitly")
    A = P.parse_args()
    require_target(P, A)
    films = {}
    for p in sorted(glob.glob(f"{ROOT}/exports/*/probe.json")):
        pb = json.load(open(p))
        slug = export_slug(pb, p)
        films[slug] = os.path.basename(pb["dir"])
    try:
        films = select_films(films, A)
    except ValueError as exc:
        P.error(str(exc))
    # Skip completed annotations unless --redo explicitly requests regeneration.
    if not A.redo:
        skip = [k for k in films
                if json.load(open(f"{ROOT}/exports/{k}/state.json")).get("script", {}).get("ok")]
        for k in skip:
            films.pop(k)
        if skip:
            print(f"跳过已完成 {len(skip)} 部", flush=True)
    if A.prep_only:
        for s in films:
            T, n = prep(s)
            print(f"  {s:<24} {n} 镜头 -> {T}/stages/script_inputs.json")
        return
    print(f"场记表队列: {len(films)} 部\n", flush=True)
    if A.workers > 1:
        print(f"并行 {A.workers} 路\n", flush=True)
        with ThreadPoolExecutor(max_workers=A.workers) as ex:
            ok = sum(bool(r) for r in ex.map(lambda kv: run(*kv, redo=A.redo), films.items()))
    else:
        ok = sum(bool(run(s, n, redo=A.redo)) for s, n in films.items())
    print(f"\n完成 {ok}/{len(films)}")
    if ok != len(films):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
