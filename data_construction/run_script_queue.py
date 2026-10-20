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
except ImportError:  # Direct script invocation.
    from release_config import (
        CODE_ROOT, DATA_ROOT, require_target, validate_data_slug, export_slug, select_films, annotation_process
    )
    from annotation_backend import build_annotation_command, AnnotationEvents

import argparse, glob, json, os, re, shutil, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

ROOT = DATA_ROOT
EFFORT = "max"   # 地基产物，后面每一步都以它为准
TOOLS = ("Read,Write,Edit,Glob,Grep,"
         "Bash(ffmpeg:*),Bash(ffprobe:*),Bash(python3:*),Bash(ls:*),Bash(cat:*),Bash(mkdir:*)")
STALL_S = 20 * 60
TRIES = 2
MAX_WAITS = 3


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
        d = json.load(open(p))
    except Exception as e:
        return [f"JSON 解析失败: {e}"]
    inp = json.load(open(f"{T}/stages/script_inputs.json"))
    want = [s["id"] for s in inp["shots"]]
    kind = {s["id"]: s["kind"] for s in inp["shots"]}
    bad = []

    got = [s.get("id") for s in d.get("shots", [])]
    if got != want:
        miss = [i for i in want if i not in got]
        extra = [i for i in got if i not in want]
        bad.append(f"镜头表对不上: {len(got)}/{len(want)} 条"
                   + (f" 缺{miss[:4]}" if miss else "") + (f" 多{extra[:4]}" if extra else "")
                   + ("" if miss or extra else " 顺序不一致"))
    for s in d.get("shots", []):
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


def _once(slug, name, T, n, attempt, why=None):
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
    open(f"{runs}/PROMPT.md", "w").write(prompt)

    t0 = time.time()
    print(f"  [{name}] {n} 镜头 启动{'（重试）' if attempt else ''} → {runs}/progress.log",
          flush=True)
    command = build_annotation_command(prompt, T, tools=TOOLS, effort=EFFORT)
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
            ok, rl, why = _once(slug, name, T, n, attempt, why)
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
