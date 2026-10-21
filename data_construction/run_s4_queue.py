"""Public construction draft: run_s4_queue.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/run_s4_queue.py [options]\n\n  --film\n  --films  Text file with one film slug per line; process only those films\n  --wait  Wait for the preprocessing queue to finish\n  -j, --workers  Number of films to process concurrently. S4 uses separate film directories;\n                         only the rate-limit window is shared; each worker waits for its reset\n  --all  Process all inputs explicitly\n  --redo  Reinspect completed S4 cuts; archive the previous draft.\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (
        CODE_ROOT, DATA_ROOT, require_target, env_text, validate_data_slug, export_slug, select_films, annotation_process
    )
    from .annotation_backend import build_annotation_command, AnnotationEvents
    from .annotation_cut_visuals import prepare_cut_visuals
    from .annotation_s4_stages import (CUT_BATCH_SIZE, INPUT_FILES, VisualReadCoverage,
        atomic_write_bytes, atomic_write_json, build_cut_prompt, make_cut_batch,
        validate_document, validate_inputs)
except ImportError:  # Direct script invocation.
    from release_config import (
        CODE_ROOT, DATA_ROOT, require_target, env_text, validate_data_slug, export_slug, select_films, annotation_process
    )
    from annotation_backend import build_annotation_command, AnnotationEvents
    from annotation_cut_visuals import prepare_cut_visuals
    from annotation_s4_stages import (CUT_BATCH_SIZE, INPUT_FILES, VisualReadCoverage,
        atomic_write_bytes, atomic_write_json, build_cut_prompt, make_cut_batch,
        validate_document, validate_inputs)

import argparse, glob, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = DATA_ROOT
EFFORT = "max"
TOOLS = ("Read,Write,Edit,Glob,Grep,"
         "Bash(ffmpeg:*),Bash(ffprobe:*),Bash(python3:*),Bash(ls:*),Bash(cat:*)")
MEASURE = re.compile(r"signalstats|blackdetect|scdet|gt\(scene|metadata=print|rawvideo|"
                     r"scenedetect|cv2|ssim|psnr|histogram", re.I)
NEED = ["cutlist", "acuts", "s3"]
STALL_S = 20 * 60        # Stop after this long without tool activity.
TRIES = 2                # Allow one retry for an intermittent failure.
MAX_WAITS = 3            # Bound the number of rate-limit waits.


def prep(slug):
    """assemble what the agent reads: cut list, units, measured audio, reference"""
    validate_data_slug(ROOT, slug)
    E, T = f"{ROOT}/exports/{slug}", f"{ROOT}/edit/task_{slug}"
    os.makedirs(f"{T}/stages", exist_ok=True); os.makedirs(f"{T}/reference", exist_ok=True)
    pb = json.load(open(f"{E}/probe.json"))
    # Absolute, and checked with lexists. A relative link stored one directory deep
    # resolves against the LINK's directory, so it dangles — and os.path.exists()
    # follows links, reports False for a dangling one, and then os.symlink raises
    # FileExistsError on the link that is sitting right there.
    g = os.path.abspath(f"{pb['dir']}/{pb['golden']}")
    if not os.path.exists(g):
        raise FileNotFoundError(f"Reference edit is missing: {g}")
    ref = f"{T}/reference/golden.mp4"
    if os.path.lexists(ref):
        cur = os.path.realpath(ref)
        if cur != g or not os.path.exists(cur):
            os.remove(ref); os.symlink(g, ref)
    else:
        os.symlink(g, ref)
    if os.path.exists(f"{T}/s3_units.json"):
        shutil.copy(f"{T}/s3_units.json", f"{T}/stages/s3_units.json")
    # the measured audio, keyed by cut index, with the golden timecodes stripped out
    ac = json.load(open(f"{E}/audio_cuts.json"))["cuts"]
    slim = [{"idx": c["idx"], "speech": c["speech"], "music": c["music"]} for c in ac]
    json.dump({"note": "measured from the finished mix — do not re-judge", "cuts": slim},
              open(f"{T}/stages/audio_measured.json", "w"), ensure_ascii=False, indent=1)

    # Use measured picture continuity; two isolated stills cannot reliably establish an unbroken shot.

    pc = f"{CODE_ROOT}/picture_cont.py"
    if os.path.exists(pc) and not os.path.exists(f"{T}/stages/picture_measured.json"):
        subprocess.run([sys.executable, "-W", "ignore", pc, "--film", slug],
                       capture_output=True, check=True)
    return T, len(slim)


def check(slug, T):
    p = f"{T}/stages/s4_cuts.json"
    if not os.path.exists(p): return ["s4_cuts.json was not generated"]
    try: d = json.load(open(p))
    except Exception as e: return [f"JSON parsing failed: {e}"]
    try:
        inputs = {key: json.loads((Path(T) / "stages" / filename).read_bytes())
                  for key, filename in INPUT_FILES.items()}
        return validate_inputs(inputs, slug) + validate_document(inputs, d)
    except (OSError, ValueError, TypeError) as exc:
        return [f"Invalid S4 inputs: {exc}"]


def _cut_phase(command, T, log_dir, coverage):
    """One isolated CLI invocation, including successful image-result coverage."""
    events, errors = AnnotationEvents(command.backend), []
    started, last = time.monotonic(), [time.monotonic()]
    stopped, killed = threading.Event(), []
    watcher, returncode, ntool = None, None, 0
    try:
        with annotation_process(command.argv, cwd=T, stderr_path=log_dir / "stderr.log") as process:
            def watch_silence():
                while not stopped.wait(30):
                    if process.poll() is not None:
                        return
                    if time.monotonic() - last[0] > STALL_S:
                        killed.append(True)
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        return
            watcher = threading.Thread(target=watch_silence, daemon=True)
            watcher.start()
            with (log_dir / "session.jsonl").open("w", encoding="utf-8") as raw, \
                 (log_dir / "progress.log").open("w", encoding="utf-8") as progress:
                count, reported, last_status = 0, 0, None
                for line in process.stdout:
                    raw.write(line); raw.flush()
                    last[0] = time.monotonic()
                    count += 1
                    for tool in events.feed(line):
                        ntool += 1
                        progress.write(f"[{last[0]-started:6.1f}s] tool: {tool['name']}\n")
                    coverage.feed(line)
                    if len(coverage.seen) != reported:
                        reported = len(coverage.seen)
                        progress.write(f"[{last[0]-started:6.1f}s] successful strip image reads "
                                       f"{reported}/{len(coverage.required)} (access coverage only)\n")
                    if last_status is None or last[0] - last_status >= 15:
                        progress.write(f"[{last[0]-started:6.1f}s] received {count} stream events\n")
                        last_status = last[0]
                    progress.flush()
            returncode = process.wait()
    except (OSError, ValueError, TimeoutError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        stopped.set()
        if watcher is not None:
            watcher.join()
    errors.extend(events.failures())
    if returncode is None and not errors:
        errors.append("Annotation process did not report exit status 0")
    if returncode is not None and returncode != 0:
        errors.insert(0, f"Annotation process exited with status {returncode}; see stderr.log")
    if killed:
        errors.insert(0, "Annotation process exceeded its inactivity limit")
    errors.extend(coverage.failures())
    return {"errors": errors, "completed": events.completed, "tools": ntool,
            "seconds": round(time.monotonic() - started, 3), "cost_usd": events.cost,
            "rate_limit": events.rate_limit, "strip_reads": len(coverage.seen),
            "required_strips": len(coverage.required), "log_dir": str(log_dir)}


def _claude_once(slug, name, T, n, runs, st, mcp, why=None, resume=True):
    run_dir = Path(tempfile.mkdtemp(prefix="batched-", dir=runs))
    source_paths = {key: Path(T) / "stages" / filename for key, filename in INPUT_FILES.items()}
    source_bytes = {path: path.read_bytes() for path in source_paths.values()}
    inputs = {key: json.loads(source_bytes[path]) for key, path in source_paths.items()}
    final_path = Path(T) / "stages/s4_cuts.json"
    original_final = final_path.read_bytes() if final_path.exists() else None
    errors, phases, published = validate_inputs(inputs, slug), {}, False
    started, visuals, draft_path, trusted = time.monotonic(), None, None, None
    model = None

    def unchanged(path, expected):
        try:
            return Path(path).read_bytes() == expected
        except OSError:
            return False

    def guard():
        bad = [f"Measured input {path.name} was changed or removed" for path, data in source_bytes.items()
               if not unchanged(path, data)]
        current = final_path.read_bytes() if final_path.exists() else None
        if current != original_final:
            bad.append("Model wrote s4_cuts.json before validated Python publication")
        if trusted is not None and not unchanged(draft_path, trusted):
            bad.append("Previously validated S4 prefix was changed or removed")
        return bad

    try:
        if not errors:
            visuals = prepare_cut_visuals(T)
            errors.extend(guard())
            draft_path = Path(visuals["draft_path"])
            if not draft_path.resolve().is_relative_to(Path(T).resolve()):
                raise ValueError("S4 working outputs must remain inside the task directory")
            draft = {"stage": "S4", "cuts": []}
            if not resume and draft_path.exists():
                shutil.copyfile(draft_path, run_dir / "previous-draft.json")
                draft_path.unlink()
            if resume and draft_path.exists():
                candidate = json.loads(draft_path.read_bytes())
                prefix_errors = validate_document(inputs, candidate, prefix=True, diversity=False)
                if prefix_errors:
                    errors.extend([f"Cannot resume invalid S4 prefix: {error}" for error in prefix_errors])
                else:
                    draft = candidate
                    phases["resume"] = {"resumed": True, "cuts": len(draft["cuts"])}
                    print(f"  [{name}] resuming validated S4 prefix {len(draft['cuts'])}/{n}; "
                          "contract checks do not certify visual accuracy", flush=True)
        if not errors:
            working = draft_path.parent / f"attempt-{run_dir.name}"
            working.mkdir()
            if not draft_path.exists():
                atomic_write_json(draft_path, draft)
            trusted = draft_path.read_bytes()
            manifest = json.loads(Path(visuals["manifest_path"]).read_bytes())
            template = (Path(CODE_ROOT) / "prompts/PROMPT_S4.md").read_text()
            # Keep shared vocabulary headings/tables, omitting the whole-film output instruction.
            vocabulary = template[template.index("### Transition vocabulary"):template.index("## Review")]
            while len(draft["cuts"]) < len(inputs["cut_list"]["cuts"]) and not errors:
                start = len(draft["cuts"])
                stop = min(len(inputs["cut_list"]["cuts"]), (start // CUT_BATCH_SIZE + 1) * CUT_BATCH_SIZE)
                batch_name = f"cuts-{start // CUT_BATCH_SIZE + 1:03d}"
                batch_dir, log_dir = working / batch_name, run_dir / batch_name
                batch_dir.mkdir(); log_dir.mkdir()
                batch, batch_manifest = make_cut_batch(inputs, manifest, start, stop)
                frozen = {}
                for key, filename in INPUT_FILES.items():
                    path = batch_dir / filename
                    atomic_write_json(path, batch[key]); frozen[path] = path.read_bytes()
                relative = lambda path: path.resolve().relative_to(Path(T).resolve()).as_posix()
                batch_manifest["inputs_paths"] = {filename: relative(batch_dir / filename)
                                                   for filename in INPUT_FILES.values()}
                batch_manifest["draft_path"] = relative(batch_dir / "cuts.json")
                manifest_path = batch_dir / "manifest.json"
                atomic_write_json(manifest_path, batch_manifest)
                frozen[manifest_path] = manifest_path.read_bytes()
                output_path = batch_dir / "cuts.json"
                prompt = build_cut_prompt(batch_dir, output_path, vocabulary)
                if why:
                    prompt += "\nPrevious attempt was rejected:\n" + "\n".join(f"- {str(e)[:500]}" for e in why[:5])
                (log_dir / "PROMPT.md").write_text(prompt, encoding="utf-8")
                command = build_annotation_command(prompt, T, tools=TOOLS, effort=EFFORT, mcp_config=mcp)
                if command.backend != "claude":
                    raise ValueError("Annotation backend changed during bounded Claude S4")
                model = command.model
                coverage = VisualReadCoverage(T, [c["strip_path"] for c in batch_manifest["cuts"]])
                print(f"  [{name}] S4 {batch_name} → {log_dir}/progress.log", flush=True)
                phase = _cut_phase(command, T, log_dir, coverage)
                phases[batch_name] = phase
                errors.extend(phase["errors"])
                errors.extend(guard())
                errors.extend(f"Frozen batch input {path.name} was changed or removed"
                              for path, data in frozen.items() if not unchanged(path, data))
                if output_path.exists():
                    shutil.copyfile(output_path, log_dir / "cuts.json")
                if not errors:
                    try:
                        annotation = json.loads(output_path.read_bytes())
                        errors.extend(validate_document(batch, annotation, diversity=False))
                    except (OSError, ValueError, TypeError) as exc:
                        errors.append(f"S4 batch output missing or invalid: {exc}")
                if not errors:
                    draft = {"stage": "S4", "cuts": draft["cuts"] + annotation["cuts"]}
                    errors.extend(validate_document(inputs, draft, prefix=True, diversity=False))
                    if not errors:
                        atomic_write_json(draft_path, draft)
                        trusted = draft_path.read_bytes()
                        print(f"  [{name}] validated S4 prefix {len(draft['cuts'])}/{n} "
                              "(data contract and image access only)", flush=True)
            if not errors:
                errors.extend(guard())
                errors.extend(validate_document(inputs, draft))
            if not errors:
                atomic_write_json(final_path, {"stage": "S4", "cuts": draft["cuts"]})
                published = True
                shutil.copyfile(final_path, run_dir / "s4_cuts.json")
                report = subprocess.run([sys.executable, "-W", "ignore", f"{CODE_ROOT}/build_report.py",
                                         "--film", slug], capture_output=True, text=True)
                if report.returncode:
                    errors.append(f"Report generation failed: {report.stderr.strip()[-160:]}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        for path, data in source_bytes.items():
            if not unchanged(path, data):
                if path.exists():
                    shutil.copyfile(path, run_dir / f"modified-{path.name}")
                atomic_write_bytes(path, data)
        if trusted is not None and not unchanged(draft_path, trusted):
            if draft_path.exists():
                shutil.copyfile(draft_path, run_dir / "modified-draft.json")
            atomic_write_bytes(draft_path, trusted)
        if not published:
            current = final_path.read_bytes() if final_path.exists() else None
            if current != original_final:
                if original_final is None:
                    final_path.unlink(missing_ok=True)
                else:
                    atomic_write_bytes(final_path, original_final)
    executed = [phase for phase in phases.values() if not phase.get("resumed")]
    costs = [phase["cost_usd"] for phase in executed]
    rate_limit = next((p["rate_limit"] for p in reversed(executed) if p.get("rate_limit")), {})
    measured = sum(_measured_ref(str(Path(p["log_dir"]) / "session.jsonl")) for p in executed
                   if (Path(p["log_dir"]) / "session.jsonl").exists())
    state = {"ok": not errors and published, "at": time.strftime("%F %T"),
             "min": round((time.monotonic()-started)/60, 1), "backend": "claude", "model": model,
             "cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
             "tools": sum(p["tools"] for p in executed), "errors": errors[:5],
             "measured_ref": measured, "rate_limit": rate_limit.get("status"),
             "phases": phases, "run_dir": str(run_dir)}
    if visuals:
        state.update(visual_manifest=str(visuals["manifest_path"]), draft=str(visuals["draft_path"]))
    st["s4"] = state
    atomic_write_json(Path(ROOT) / "exports" / slug / "state.json", st)
    print(f"  [{name}] " + ("✓ S4 batches validated" if state["ok"] else f"✗ {(errors or ['No final output'])[0]}"), flush=True)
    return state["ok"], rate_limit, errors


def _once(slug, name, T, n, attempt, why=None, resume=True):
    validate_data_slug(ROOT, slug)
    mcp = env_text("VIDEO_EDITING_MCP_CONFIG")
    E = f"{ROOT}/exports/{slug}"
    st = json.load(open(f"{E}/state.json"))
    st["s4"] = {"ok": False, "at": time.strftime("%F %T"), "errors": ["Annotation attempt incomplete"]}
    json.dump(st, open(f"{E}/state.json", "w"), ensure_ascii=False, indent=1)
    runs = f"{ROOT}/edit/runs/{slug}/{time.strftime('%Y%m%d_%H%M%S')}_s4"
    os.makedirs(runs, exist_ok=True)

    # check() only looks at the file, and a leftover s4_cuts.json from an earlier
    # session validates just fine — an agent that dies on startup would be recorded
    # as a success against yesterday's output. Require this run to have touched it.
    out_p = f"{T}/stages/s4_cuts.json"
    pre_m = os.path.getmtime(out_p) if os.path.exists(out_p) else 0.0
    prompt = open(f"{CODE_ROOT}/prompts/PROMPT_S4.md").read()
    prompt += f"\n\n## Current film\n\n{name}: **{n} cut boundaries**.\n"
    if why:
        # An identical retry repeats the identical failure. Tell it what was rejected.
        prompt += ("\n## Previous attempt failed validation\n\nReasons:\n- " + "\n- ".join(why[:5]) +
                   "\n\nAddress these failures. Submitting unchanged preliminary placeholders is incomplete;"
                   " inspect the pictures at every boundary and ground each technique label in the evidence.\n")
    open(f"{runs}/PROMPT.md", "w").write(prompt)

    t0 = time.time()
    print(f"  [{name}] Starting {n} cut boundaries{' (retry)' if attempt else ''} → {runs}/progress.log",
          flush=True)
    command = build_annotation_command(prompt, T, tools=TOOLS, effort=EFFORT, mcp_config=mcp)
    if command.backend == "claude":
        return _claude_once(slug, name, T, n, runs, st, mcp, why=why, resume=resume)
    events = AnnotationEvents(command.backend)
    with annotation_process(command.argv, cwd=T, stderr_path=f"{runs}/stderr.log") as p:

        # A stalled agent used to block the queue for good: the read loop below simply
        # waits, and nothing else in this script has a clock. Watch for silence rather
        # than total runtime — a slow film is fine, a mute one is not.
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
    if killed: print(f"  [{name}] Stopped after more than {STALL_S//60} minutes without activity", flush=True)
    dt = time.time() - t0

    bad = events.failures() + check(slug, T)
    if returncode != 0:
        bad.insert(0, f"Annotation process exited with status {returncode}; see stderr.log")
    if killed:
        bad.insert(0, "Annotation process exceeded its inactivity limit")
    if not bad and (not os.path.exists(out_p) or os.path.getmtime(out_p) <= pre_m):
        bad = ["s4_cuts.json was not written by this attempt; it is left over from a previous run"]
    meas = _measured_ref(f"{runs}/session.jsonl", command.backend)
    if os.path.exists(f"{T}/stages/s4_cuts.json"):
        shutil.copy(f"{T}/stages/s4_cuts.json", f"{runs}/s4_cuts.json")
    st["s4"] = {"ok": not bad, "at": time.strftime("%F %T"), "min": round(dt / 60, 1),
                "backend": command.backend, "model": command.model,
                "cost_usd": cost, "tools": ntool, "errors": bad[:5], "measured_ref": meas,
                "rate_limit": rl[0].get("status")}
    if not bad:
        # returncode checked: this used to be fire-and-forget, so a report that failed
        # to build looked exactly like one that built fine.
        rb = subprocess.run([sys.executable, "-W", "ignore", f"{CODE_ROOT}/build_report.py",
                             "--film", slug], capture_output=True, text=True)
        if rb.returncode:
            print(f"  [{name}] ⚠ Report generation failed: {rb.stderr.strip()[-200:]}", flush=True)
            bad = bad + [f"Report generation failed: {rb.stderr.strip()[-160:]}"]
    st["s4"].update(ok=not bad, errors=bad[:5])
    json.dump(st, open(f"{E}/state.json", "w"), ensure_ascii=False, indent=1)
    tag = "✓" if not bad else f"✗ {bad[0][:60]}"
    print(f"  [{name}] {dt/60:.1f} min, {ntool} tool calls" + (f" ${cost:.2f}" if cost else "") +
          f"  {tag}" + (f"  reference measurement issue: {meas}" if meas else ""), flush=True)
    return (not bad, rl[0], bad)


def _measured_ref(path, backend="claude"):
    """Count reference measurement commands, excluding text written in heredocs."""
    n = 0
    events = AnnotationEvents(backend)
    with open(path) as stream:
        for line in stream:
            for tool in events.feed(line):
                if tool["name"] != "Bash":
                    continue
                cmd = str(tool["input"].get("command", ""))
                cmd = re.sub(r"<<\s*'?(\w+)'?[\s\S]*?^\1", "", cmd, flags=re.M)
                if MEASURE.search(cmd) and re.search(r"reference/golden|golden\.mp4", cmd):
                    n += 1
    return n


def run(slug, name, redo=False):
    """Process one film without propagating its failure to the other queue items."""
    validate_data_slug(ROOT, slug)
    try:
        return _film(slug, name, redo=redo)
    except Exception as e:
        print(f"  [{name}] ✗ Unhandled exception: {type(e).__name__}: {str(e)[:160]}", flush=True)
        try:
            sf = f"{ROOT}/exports/{slug}/state.json"
            st = json.load(open(sf))
            st["s4"] = {"ok": False, "at": time.strftime("%F %T"),
                        "errors": [f"{type(e).__name__}: {str(e)[:200]}"]}
            json.dump(st, open(sf, "w"), ensure_ascii=False, indent=1)
        except (OSError, ValueError) as state_error:
            print(f"  [{name}] Could not record failure state: {state_error}", file=sys.stderr, flush=True)
        return False


def _film(slug, name, redo=False):
    E = f"{ROOT}/exports/{slug}"
    st = json.load(open(f"{E}/state.json"))
    miss = [s for s in NEED if not st.get(s, {}).get("ok")]
    if miss:
        print(f"  Skipping {name}: preprocessing is incomplete {miss}", flush=True); return False
    if st.get("s4", {}).get("ok") and not redo:
        print(f"  Skipping {name}: S4 is complete", flush=True); return True
    T, n = prep(slug)
    attempt = waits = 0
    why = None
    while attempt < TRIES:
        ok, rl, why = _once(slug, name, T, n, attempt, why,
                            resume=not redo or attempt > 0 or waits > 0)
        if ok: return True
        # A rate-limited run is not this film's failure — it is a global stop. Left
        # alone, the queue would burn its two tries here and then march through every
        # remaining film the same way, marking the lot failed inside a few minutes.
        if rl.get("status") not in (None, "allowed") and waits < MAX_WAITS:
            wake = (rl.get("resetsAt") or (time.time() + 900)) + 60
            print(f"  [{name}] Rate limited ({rl.get('rateLimitType')}); retry at "
                  f"{time.strftime('%H:%M', time.localtime(wake))}", flush=True)
            while time.time() < wake: time.sleep(60)
            waits += 1
            continue                  # Rate-limit waits do not consume an attempt.
        attempt += 1
    return False


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--film")
    P.add_argument("--redo", action="store_true", help="Reinspect completed S4 cuts; archive the previous draft")
    # A film list limits the selected inputs while allowing independent per-film jobs to run concurrently.

    P.add_argument("--films", help="Text file with one film slug per line; process only those films")
    P.add_argument("--wait", action="store_true",
                                             help="Wait for the preprocessing queue to finish")
    P.add_argument("-j", "--workers", type=int, default=1,
                   help="Number of films to process concurrently. S4 uses separate film directories;\n                         only the rate-limit window is shared; each worker waits for its reset")
    P.add_argument("--all", action="store_true", help="Process all inputs explicitly")
    A = P.parse_args()
    require_target(P, A)
    if A.wait:
        # Any pipeline run, not just --all. A targeted `--film X --only s3,...` was
        # invisible to the old pattern, so S4 could start while a film's s3 was still
        # missing — and a film skipped for missing artifacts is skipped for good.
        while True:
            result = subprocess.run(["pgrep", "-f", "pipeline.py"], capture_output=True)
            if result.returncode == 1:
                break
            if result.returncode != 0:
                raise RuntimeError(f"Cannot check pipeline status: pgrep exited {result.returncode}")
            time.sleep(120)
        print("Preprocessing queue finished; starting S4\n", flush=True)

    films = {}
    for p in sorted(glob.glob(f"{ROOT}/exports/*/probe.json")):
        pb = json.load(open(p))
        slug = export_slug(pb, p)
        films[slug] = os.path.basename(pb["dir"])
    try:
        films = select_films(films, A)
    except ValueError as exc:
        P.error(str(exc))
    print(f"S4 queue: {len(films)} films\n", flush=True)
    if A.workers > 1:
        print(f"Using {A.workers} workers (separate film files; shared rate-limit window)\n", flush=True)
        with ThreadPoolExecutor(max_workers=A.workers) as ex:
            ok = sum(bool(r) for r in ex.map(lambda kv: run(*kv, redo=A.redo), films.items()))
    else:
        ok = sum(bool(run(slug, name, redo=A.redo)) for slug, name in films.items())
    print(f"\nS4 completed {ok}/{len(films)} films")
    _clear_sentinel(films)          # Clear the sentinel only after every pending film has been reinspected.
    if ok != len(films):
        raise SystemExit(1)


def _clear_sentinel(done_films):
    """drop the pending marker once every film it named has S4 output again"""
    p = f"{ROOT}/edit/S4_PENDING.json"
    if not os.path.exists(p):
        return
    want = json.load(open(p)).get("films") or []
    left = []
    for slug in want:
        validate_data_slug(ROOT, slug)
        try:
            state = json.load(open(f"{ROOT}/exports/{slug}/state.json"))
        except (OSError, ValueError):
            state = {}
        if (not state.get("s4", {}).get("ok")
                or not os.path.exists(f"{ROOT}/edit/task_{slug}/stages/s4_cuts.json")):
            left.append(slug)
    if left:
        print(f"  · {len(left)} films still need reinspection; retaining sentinel: {left[:4]}", flush=True)
    else:
        os.remove(p)
        print("  · Reinspection complete; cleared edit/S4_PENDING.json", flush=True)


if __name__ == "__main__":
    main()
