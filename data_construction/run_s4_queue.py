"""Public construction draft: run_s4_queue.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/run_s4_queue.py [options]\n\n  --film\n  --films  文本文件，每行一个 slug；只跑这些\n  --wait  等机械队列跑完再开始\n  -j, --workers  同时跑几部片。S4 不碰 Resolve/Modal，各片只写自己的目录，\n                         唯一共享的是速率窗口 —— 撞上会各自等到重置\n  --all  Process all inputs explicitly\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (
        CODE_ROOT, DATA_ROOT, require_target, require_annotation_model, env_text, validate_data_slug, export_slug, select_films, annotation_cli, annotation_process
    )
except ImportError:  # Direct script invocation.
    from release_config import (
        CODE_ROOT, DATA_ROOT, require_target, require_annotation_model, env_text, validate_data_slug, export_slug, select_films, annotation_cli, annotation_process
    )

import argparse, glob, json, os, re, shutil, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

ROOT = DATA_ROOT
EFFORT = "max"
TOOLS = ("Read,Write,Edit,Glob,Grep,"
         "Bash(ffmpeg:*),Bash(ffprobe:*),Bash(python3:*),Bash(ls:*),Bash(cat:*)")
DENY = "Agent,Task"
MEASURE = re.compile(r"signalstats|blackdetect|scdet|gt\(scene|metadata=print|rawvideo|"
                     r"scenedetect|cv2|ssim|psnr|histogram", re.I)
NEED = ["cutlist", "acuts", "s3"]
STALL_S = 20 * 60        # 没有任何工具调用超过这么久 = 卡死
TRIES = 2                # 一次偶发失败不该永久丢掉一部片
MAX_WAITS = 3            # 最多为限流等三次窗口，避免无限挂着


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
        raise FileNotFoundError(f"参考带不存在: {g}")
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
    if not os.path.exists(p): return ["s4_cuts.json 未生成"]
    try: d = json.load(open(p))
    except Exception as e: return [f"JSON 解析失败: {e}"]
    cl = json.load(open(f"{T}/stages/cut_list.json"))["cuts"]
    cu = d.get("cuts", []); bad = []
    if len(cu) != len(cl): bad.append(f"切点数 {len(cu)} != {len(cl)}")
    for i, (a, b) in enumerate(zip(cu, cl)):
        if (a.get("after"), a.get("before")) != (b["after"], b["before"]):
            bad.append(f"第 {i+1} 条相邻关系不对"); break
    for c in cu:
        if not (c.get("transition") or {}).get("type"): bad.append(f"#{c.get('idx')} 缺 transition")
        if not c.get("technique"): bad.append(f"#{c.get('idx')} 缺 technique")
        if not c.get("note"): bad.append(f"#{c.get('idx')} 缺 note")

    techs = {c.get("technique") for c in cu if c.get("technique")}
    if len(cu) >= 20 and len(techs) < 3:
        bad.append(f"只有 {len(techs)} 种手法 {sorted(techs)} —— 这是初判占位表，没有回头精修")
    return bad


def _once(slug, name, T, n, attempt, why=None):
    validate_data_slug(ROOT, slug)
    model = require_annotation_model()
    cli = annotation_cli()
    mcp = env_text("VIDEO_EDITING_MCP_CONFIG")
    if mcp:
        mcp = os.path.abspath(os.path.expanduser(mcp))
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
    prompt += f"\n\n## 本片\n\n《{name}》，共 **{n} 个切点**。\n"
    if why:
        # An identical retry repeats the identical failure. Tell it what was rejected.
        prompt += ("\n## 上一次这部片被打回了\n\n原因：\n- " + "\n- ".join(why[:5]) +
                   "\n\n这次务必避开。特别注意：把初判占位表原样交上去不算完成，"
                   "必须逐刀回头看画面、给出真实的手法判断。\n")
    open(f"{runs}/PROMPT.md", "w").write(prompt)

    t0 = time.time()
    print(f"  [{name}] {n} 切点 启动{'（重试）' if attempt else ''} → {runs}/progress.log",
          flush=True)
    with annotation_process(
        [cli, "-p", prompt, "--model", model, "--effort", EFFORT,
         "--allowedTools", TOOLS, "--disallowedTools", DENY,
         "--permission-mode", "acceptEdits",
         *(["--mcp-config", mcp, "--strict-mcp-config"] if mcp else []),
         "--add-dir", T, "--output-format", "stream-json", "--verbose"],
        cwd=T, stderr_path=f"{runs}/stderr.log") as p:

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
            ntool, cost, rl = 0, None, [{}]
            for line in p.stdout:
                raw.write(line); raw.flush()
                last[0] = time.time()
                try: ev = json.loads(line)
                except Exception: continue
                if ev.get("type") == "assistant":
                    for c in ev.get("message", {}).get("content", []):
                        if c.get("type") == "tool_use":
                            ntool += 1
                            a = c["input"].get("file_path") or c["input"].get("command") or ""
                            one = " ".join(str(a).split())
                            # Preserve the end of long paths because it distinguishes individual inspection artifacts.


                            one = ("…" + one[-72:]) if (len(one) > 74 and "/" in one) else one[:80]
                            prog.write(f"[{time.time()-t0:6.0f}s] → {c['name']}  {one}\n")
                            prog.flush()
                if ev.get("type") == "rate_limit_event":
                    rl[0] = ev.get("rate_limit_info") or {}
                if ev.get("type") == "result": cost = ev.get("total_cost_usd")
        returncode = p.wait()
    if killed: print(f"  [{name}] ⚠ 静默超过 {STALL_S//60} 分钟，已终止", flush=True)
    dt = time.time() - t0

    bad = check(slug, T)
    if returncode != 0:
        bad.insert(0, f"Annotation process exited with status {returncode}; see stderr.log")
    if killed:
        bad.insert(0, "Annotation process exceeded its inactivity limit")
    if not bad and (not os.path.exists(out_p) or os.path.getmtime(out_p) <= pre_m):
        bad = ["s4_cuts.json 没被本次运行写过 —— 是上次遗留的文件"]
    meas = _measured_ref(f"{runs}/session.jsonl")
    if os.path.exists(f"{T}/stages/s4_cuts.json"):
        shutil.copy(f"{T}/stages/s4_cuts.json", f"{runs}/s4_cuts.json")
    st["s4"] = {"ok": not bad, "at": time.strftime("%F %T"), "min": round(dt / 60, 1),
                "cost_usd": cost, "tools": ntool, "errors": bad[:5], "measured_ref": meas,
                "rate_limit": rl[0].get("status")}
    if not bad:
        # returncode checked: this used to be fire-and-forget, so a report that failed
        # to build looked exactly like one that built fine.
        rb = subprocess.run([sys.executable, "-W", "ignore", f"{CODE_ROOT}/build_report.py",
                             "--film", slug], capture_output=True, text=True)
        if rb.returncode:
            print(f"  [{name}] ⚠ 报告生成失败: {rb.stderr.strip()[-200:]}", flush=True)
            bad = bad + [f"报告生成失败: {rb.stderr.strip()[-160:]}"]
    st["s4"].update(ok=not bad, errors=bad[:5])
    json.dump(st, open(f"{E}/state.json", "w"), ensure_ascii=False, indent=1)
    tag = "✓" if not bad else f"✗ {bad[0][:60]}"
    print(f"  [{name}] {dt/60:.1f}分 {ntool}次" + (f" ${cost:.2f}" if cost else "") +
          f"  {tag}" + (f"  ⚠测量参考带 {meas}" if meas else ""), flush=True)
    return (not bad, rl[0], bad)


def _measured_ref(path):
    """Count measurement commands executed against the reference.
    Inspect Bash tool calls and strip heredoc bodies so written text is not counted as execution."""
    n = 0
    for l in open(path):
        try: ev = json.loads(l)
        except Exception: continue
        if ev.get("type") != "assistant": continue
        for c in ev.get("message", {}).get("content", []):
            if c.get("type") != "tool_use" or c.get("name") != "Bash": continue
            cmd = str(c.get("input", {}).get("command", ""))
            cmd = re.sub(r"<<\s*'?(\w+)'?[\s\S]*?^\1", "", cmd, flags=re.M)   # 去掉 heredoc 正文
            if MEASURE.search(cmd) and re.search(r"reference/golden|golden\.mp4", cmd):
                n += 1
    return n


def run(slug, name):
    """Process one film without propagating its failure to the other queue items."""
    validate_data_slug(ROOT, slug)
    try:
        return _film(slug, name)
    except Exception as e:
        print(f"  [{name}] ✗ 未捕获异常: {type(e).__name__}: {str(e)[:160]}", flush=True)
        try:
            sf = f"{ROOT}/exports/{slug}/state.json"
            st = json.load(open(sf))
            st["s4"] = {"ok": False, "at": time.strftime("%F %T"),
                        "errors": [f"{type(e).__name__}: {str(e)[:200]}"]}
            json.dump(st, open(sf, "w"), ensure_ascii=False, indent=1)
        except (OSError, ValueError) as state_error:
            print(f"  [{name}] 无法记录失败状态: {state_error}", file=sys.stderr, flush=True)
        return False


def _film(slug, name):
    E = f"{ROOT}/exports/{slug}"
    st = json.load(open(f"{E}/state.json"))
    miss = [s for s in NEED if not st.get(s, {}).get("ok")]
    if miss:
        print(f"  跳过 {name}: 机械阶段未完成 {miss}", flush=True); return False
    if st.get("s4", {}).get("ok"):
        print(f"  跳过 {name}: S4 已完成", flush=True); return True
    T, n = prep(slug)
    attempt = waits = 0
    why = None
    while attempt < TRIES:
        ok, rl, why = _once(slug, name, T, n, attempt, why)
        if ok: return True
        # A rate-limited run is not this film's failure — it is a global stop. Left
        # alone, the queue would burn its two tries here and then march through every
        # remaining film the same way, marking the lot failed inside a few minutes.
        if rl.get("status") not in (None, "allowed") and waits < MAX_WAITS:
            wake = (rl.get("resetsAt") or (time.time() + 900)) + 60
            print(f"  [{name}] 触到 {rl.get('rateLimitType')} 限流，等到 "
                  f"{time.strftime('%H:%M', time.localtime(wake))} 再试", flush=True)
            while time.time() < wake: time.sleep(60)
            waits += 1
            continue                  # 不消耗重试次数
        attempt += 1
    return False


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--film")
    # A film list limits the selected inputs while allowing independent per-film jobs to run concurrently.

    P.add_argument("--films", help="文本文件，每行一个 slug；只跑这些")
    P.add_argument("--wait", action="store_true",
                                             help="等机械队列跑完再开始")
    P.add_argument("-j", "--workers", type=int, default=1,
                   help="同时跑几部片。S4 不碰 Resolve/Modal，各片只写自己的目录，\n                         唯一共享的是速率窗口 —— 撞上会各自等到重置")
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
        print("机械队列已结束，开始 S4\n", flush=True)

    films = {}
    for p in sorted(glob.glob(f"{ROOT}/exports/*/probe.json")):
        pb = json.load(open(p))
        slug = export_slug(pb, p)
        films[slug] = os.path.basename(pb["dir"])
    try:
        films = select_films(films, A)
    except ValueError as exc:
        P.error(str(exc))
    print(f"S4 队列: {len(films)} 部\n", flush=True)
    if A.workers > 1:
        print(f"并行 {A.workers} 路（各片互不共享文件；共享的只有速率窗口）\n", flush=True)
        with ThreadPoolExecutor(max_workers=A.workers) as ex:
            ok = sum(bool(r) for r in ex.map(lambda kv: run(*kv), films.items()))
    else:
        ok = sum(bool(run(slug, name)) for slug, name in films.items())
    print(f"\nS4 完成 {ok}/{len(films)}")
    _clear_sentinel(films)          # 全部重判完才清；否则哨兵留着
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
        print(f"  · 仍有 {len(left)} 部未重判，哨兵保留: {left[:4]}", flush=True)
    else:
        os.remove(p)
        print("  · 全部重判完成，已清除 edit/S4_PENDING.json", flush=True)


if __name__ == "__main__":
    main()
