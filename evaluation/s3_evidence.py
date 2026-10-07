"""Extract the evidence three S3 rubrics need, before any model looks at the work.

The split is deliberate: everything a machine can establish is established here, once, and
written down. The model that scores later reads a page of measurements instead of poking at
media — which makes the call reproducible, cheap, and arguable. When a score looks wrong you
can open the evidence and see whether the measurement was wrong or the judgement was.

Three rubrics, three bundles:

    placement   where each in-point and out-point landed — mid-word or in the clear,
                mid-gesture or after the action read
    coverage    what the assembled cut still contains of the beat the brief asked for,
                and — the part that matters — what got trimmed away
    jumpcut     how alike the two frames on either side of each join are, measured
                against how alike those two clips are anyway

The last one needs that baseline. Two clips from the same setup look alike no matter where
you trim them; that is the material, not the edit. What the editor controls is whether the
chosen frames make it worse than it had to be.

    python3 s3_evidence.py --task path/to/task
    python3 s3_evidence.py --tasks /tmp/s3_sample.json
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile, warnings
from grading_video import clamped_time, frame_in_span, s3_videos, VIDEO_CONTRACT

if __name__ != "__main__":
    raise ImportError(f"{__file__} 是命令行脚本，不能被 import（会写文件、起子进程）")

P = argparse.ArgumentParser()
P.add_argument("--task")
P.add_argument("--tasks", help="json 数组，里面是任务目录")
P.add_argument("--asr", default="small.en", help="转录模型")
P.add_argument("--force", action="store_true")
A = P.parse_args()

warnings.filterwarnings("ignore")
_ASR = []
ASR_CONTRACT = "s12-asr-v3-vad500-noprev-canned-filter"
_CANNED_HALLUCINATIONS = {
    "thanks for watching", "thank you for watching", "please subscribe",
    "like and subscribe", "subtitles by the amara org community",
}
_HEALTH = {
    "format": "s12-evidence-health-v3",
    "asr": {
        "status": "ok",
        "model": A.asr,
        "contract": ASR_CONTRACT,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "cache_hits": 0,
        "shared_cache_hits": 0,
        "shared_cache_writes": 0,
        "runtime_transcriptions": 0,
        "empty_results": 0,
        "filtered_hallucination_segments": 0,
        "legacy_cache_bypasses": 0,
        "cache_errors": [],
    },
}


def _bind_evidence_slot():
    """Keep one evidence process and all of its FFmpeg children CPU-bounded."""
    raw = os.environ.get("S12_EVIDENCE_SLOT")
    if raw is None or not hasattr(os, "sched_getaffinity"):
        return
    try:
        allowed = sorted(os.sched_getaffinity(0))
        per_slot = max(1, int(os.environ.get("S12_EVIDENCE_CPUS_PER_SLOT", "2")))
        start = (int(raw) * per_slot) % len(allowed)
        selected = {allowed[(start + i) % len(allowed)] for i in range(per_slot)}
        os.sched_setaffinity(0, selected)
    except (OSError, TypeError, ValueError, ZeroDivisionError):
        # Affinity is an optimisation.  The process-shared gate remains the
        # correctness boundary on kernels/runtimes that do not expose it.
        return


_bind_evidence_slot()


def sh(cmd, binary=False):
    return subprocess.run(cmd, capture_output=True, text=not binary)


def probe_dur(p):
    r = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", p]).stdout.strip()
    return float(r) if r else 0.0


def _is_canned_hallucination(segment) -> bool:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(getattr(segment, "text", "")).lower())
    text = " ".join(text.split())
    try:
        no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
    except (TypeError, ValueError):
        no_speech = 0.0
    return text in _CANNED_HALLUCINATIONS and no_speech >= 0.20


def _publish_shared_words_cache(path, words):
    """Best-effort acceleration; cache I/O must not invalidate real ASR evidence."""
    tmp = None
    fd = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".",
            suffix=".tmp", dir=os.path.dirname(path),
        )
        handle = os.fdopen(fd, "w", encoding="utf-8")
        fd = None  # The file object now owns the descriptor, including on error.
        with handle:
            json.dump({"contract": ASR_CONTRACT, "words": words}, handle,
                      ensure_ascii=False)
        os.replace(tmp, path)
        tmp = None
        _HEALTH["asr"]["shared_cache_writes"] += 1
    except OSError as exc:
        # Concurrent Modal Volume writers can reject replace with ETXTBSY.
        # Keep the successful transcription; do not rerun the entire judge or
        # substitute an empty transcript. Actual ASR failures remain fatal.
        _HEALTH["asr"]["cache_errors"].append(
            f"shared publish/{os.path.basename(path)}: {type(exc).__name__}: {exc}"
        )
    finally:
        if fd is not None:
            os.close(fd)
        if tmp is not None:
            try:
                os.unlink(tmp)  # Only our unique temporary file, never the winner.
            except FileNotFoundError:
                pass
            except OSError as exc:
                _HEALTH["asr"]["cache_errors"].append(
                    f"shared cleanup/{os.path.basename(tmp)}: {type(exc).__name__}: {exc}"
                )


def words_of(path):
    """整条 clip 的词级转录。切点落没落在词中间，只有这个能回答。

    优先读预计算缓存 `<clip>.asr.json` —— ASR 只依赖 clip 本身、和模型输出无关，
    所以离线算一次就够。放在 rollout 里现算的代价太大：每条 s_3 轨迹 9 个 clip，
    CPU 上 small.en 一个约 10 秒，等于给每条轨迹白加 90 秒。

    缓存缺失时必须真正运行 ASR。依赖、模型初始化或转录失败都抛错，绝不能
    伪装成“这条素材没有对白”；无词只有在一次成功转录后才是有效测量。
    """
    from grading_contract import media_info
    if not any(s["codec_type"] == "audio" for s in media_info(path)["streams"]):
        # A genuinely silent video has no ASR input; do not confuse this with an
        # ASR dependency/decoder failure or reuse stale speech from a sidecar.
        _HEALTH["asr"]["no_audio_stream"] = _HEALTH["asr"].get("no_audio_stream", 0) + 1
        return []
    cache = os.path.splitext(path)[0] + ".asr.json"
    if os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as fh:
                cached = json.load(fh)
            if not isinstance(cached, dict) or cached.get("contract") != ASR_CONTRACT:
                raise ValueError("legacy/unversioned ASR cache")
            cached = cached.get("words")
            if not isinstance(cached, list):
                raise TypeError(f"expected word list, got {type(cached).__name__}")
            _HEALTH["asr"]["cache_hits"] += 1
            if not cached:
                _HEALTH["asr"]["empty_results"] += 1
            return cached
        except Exception as exc:  # fall back to live ASR, but expose corruption
            if str(exc) == "legacy/unversioned ASR cache":
                _HEALTH["asr"]["legacy_cache_bypasses"] += 1
            else:
                _HEALTH["asr"]["cache_errors"].append(
                    f"{os.path.basename(cache)}: {type(exc).__name__}: {exc}"
                )
    shared_cache = None
    cache_root = os.environ.get("S12_ASR_CACHE_DIR", "").strip()
    real = os.path.realpath(path)
    # Remote grading reconstructs ``initial_state`` through a symlink inside a
    # private /tmp task.  Depending on the Volume mount implementation,
    # realpath may not retain the literal /data prefix.  The semantic
    # initial_state boundary is the stable distinction: cache trusted source
    # media, never a policy-produced output/preview.
    source_media = "/initial_state/" in path.replace(os.sep, "/")
    if cache_root and source_media:
        stat = os.stat(real)
        key = hashlib.sha256(
            f"s3-words\0{ASR_CONTRACT}\0{A.asr}\0{real}\0"
            f"{stat.st_size}\0{stat.st_mtime_ns}".encode()
        ).hexdigest()
        shared_cache = os.path.join(cache_root, "s3", key[:2], key + ".json")
        if os.path.isfile(shared_cache):
            try:
                with open(shared_cache, encoding="utf-8") as handle:
                    wrapper = json.load(handle)
                if (not isinstance(wrapper, dict)
                        or wrapper.get("contract") != ASR_CONTRACT):
                    raise TypeError("shared ASR cache contract mismatch")
                cached = wrapper.get("words")
                if not isinstance(cached, list):
                    raise TypeError("shared ASR cache is not a word list")
                _HEALTH["asr"]["shared_cache_hits"] += 1
                if not cached:
                    _HEALTH["asr"]["empty_results"] += 1
                return cached
            except Exception as exc:
                _HEALTH["asr"]["cache_errors"].append(
                    f"shared/{key}: {type(exc).__name__}: {exc}"
                )
    if not _ASR:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "S3 ASR unavailable: install faster-whisper in the grader image"
            ) from exc
        try:
            _ASR.append(WhisperModel(A.asr, device="cpu", compute_type="int8"))
        except Exception as exc:
            raise RuntimeError(
                f"S3 ASR model {A.asr!r} failed to initialise: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    try:
        segs, _ = _ASR[0].transcribe(
            path,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
        out = []
        # faster-whisper decodes lazily while iterating segments, so this loop
        # belongs inside the exception boundary too.
        for s in segs:
            if _is_canned_hallucination(s):
                _HEALTH["asr"]["filtered_hallucination_segments"] += 1
                continue
            for w in (s.words or []):
                if w.word.strip():
                    out.append({"w": w.word.strip(), "s": round(w.start, 3),
                                "e": round(w.end, 3)})
    except Exception as exc:
        raise RuntimeError(
            f"S3 ASR failed for {path}: {type(exc).__name__}: {exc}"
        ) from exc
    _HEALTH["asr"]["runtime_transcriptions"] += 1
    if not out:
        _HEALTH["asr"]["empty_results"] += 1
    if shared_cache:
        _publish_shared_words_cache(shared_cache, out)
    return out


def gray(path, w=48, h=27, ss=None, n=1, fps=None, bounds=None):
    """取灰度帧。所有画面度量都从这里出来，尺寸统一，量纲才可比。"""
    import numpy as np
    cmd = ["ffmpeg", "-v", "error"]
    if bounds is not None:
        chosen = frame_in_span(path, ss, *bounds)
        if chosen is None:
            return None
        cmd += ["-i", path, "-frames:v", "1", "-vf", f"select=eq(n\\,{chosen[0]}),scale={w}:{h}",
                "-pix_fmt", "gray", "-f", "rawvideo", "-"]
        result = subprocess.run(cmd, capture_output=True)
        pixels = np.frombuffer(result.stdout, dtype=np.uint8)
        return pixels.reshape(1, h, w).astype("float32") if pixels.size == w*h else None
    if ss is not None:
        cmd += ["-ss", f"{max(0.0, ss):.4f}"]
    cmd += ["-i", path]
    if fps:
        cmd += ["-vf", f"fps={fps},scale={w}:{h}"]
    else:
        cmd += ["-frames:v", str(n), "-vf", f"scale={w}:{h}"]
    cmd += ["-pix_fmt", "gray", "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    k = w * h
    return a[: (a.size // k) * k].reshape(-1, h, w).astype("float32") if a.size >= k else None


def motion_profile(path, fps=12):
    """整条 clip 的逐帧变化量。切点处的运动强度要跟这条 clip 自己比，
    不能跟全库比——手持长焦和三脚架固定机位的基线差一个数量级。"""
    import numpy as np
    fr = gray(path, fps=fps)
    if fr is None or len(fr) < 2:
        return None
    d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    return {"fps": fps, "v": [round(float(x), 2) for x in d],
            "median": round(float(np.median(d)), 2),
            "p90": round(float(np.percentile(d, 90)), 2)}


def at_motion(prof, t):
    if not prof:
        return None
    i = int(t * prof["fps"])
    v = prof["v"]
    if not v:
        return None
    i = max(0, min(i, len(v) - 1))
    return v[i]


def speech_at(words, t, edge):
    """切点跟语音的关系。edge='in' 看它切进来时人是不是正在说话。"""
    inside = [w for w in words if w["s"] - 0.02 <= t <= w["e"] + 0.02]
    if inside:
        w = inside[0]
        frac = (t - w["s"]) / max(1e-3, w["e"] - w["s"])
        return {"state": "mid_word", "word": w["w"],
                "into_word_s": round(t - w["s"], 3),
                "into_word_pct": round(100 * min(1.0, max(0.0, frac))),
                "word_span": [w["s"], w["e"]]}
    before = [w for w in words if w["e"] <= t]
    after = [w for w in words if w["s"] >= t]
    gap_b = round(t - before[-1]["e"], 3) if before else None
    gap_a = round(after[0]["s"] - t, 3) if after else None
    return {"state": "in_the_clear", "gap_before_s": gap_b, "gap_after_s": gap_a,
            "prev_word": before[-1]["w"] if before else None,
            "next_word": after[0]["w"] if after else None}


def sheet(items, out_png, cols=6, w=320):
    """一张联络表。文字量不出"这一刀看着顺不顺"，画面能。

    每格左上角压一个标签（p03 IN / j05 pre 之类），判官才能把看到的画面跟证据里的行对上。

    每帧单独抽好、统一裁到同一尺寸，再用 `tile` 拼。先前用 xstack 的 `w0*n_h0*n`
    布局，各镜头宽高比不同导致瓦片高度不齐，整张静默失败——`frames_out.png` 一直
    没生成就是这个原因。tile 要求输入等尺寸，所以 scale 后再 pad 到固定框。
    """
    if not items:
        return None
    h = int(w * 9 / 16)
    tmp = tempfile.mkdtemp(prefix="s3sheet")
    try:
        n = 0
        failures = []
        for i, item in enumerate(items):
            path, t, label = item[:3]
            bounds = item[3] if len(item) > 3 else None
            lab = str(label).replace(":", "\\:").replace("'", "")
            q = f"{tmp}/{i:03d}.png"
            vf = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
                f"drawtext=text='{lab}':x=8:y=8:fontsize=24:fontcolor=white:"
                f"box=1:boxcolor=black@0.7:boxborderw=6"
            )
            # Select an actual decoded PTS within the labelled source interval.
            # No EOF retries may cross in/out points and silently show discarded frames.
            from grading_contract import media_info
            dur = float(next(x for x in media_info(path)["streams"] if x["codec_type"] == "video")["duration"])
            lo, hi = bounds if bounds is not None else (0., dur)
            chosen = frame_in_span(path, t, lo, hi)
            if chosen is None:
                placeholder_vf = (
                    f"drawtext=text='{lab}':x=8:y=8:fontsize=24:fontcolor=white,"
                    "drawtext=text='NO FRAME IN LABELLED INTERVAL':x=5:y=60:fontsize=16:fontcolor=white")
                run = sh(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                          f"color=c=0x5a1010:s={w}x{h}:d=0.04", "-frames:v", "1",
                          "-vf", placeholder_vf, "-y", q])
            else:
                run = sh(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1",
                          "-vf", f"select=eq(n\\,{chosen[0]})," + vf, "-y", q])
            if run.returncode == 0 and os.path.isfile(q) and os.path.getsize(q) > 0:
                os.rename(q, f"{tmp}/t{n:03d}.png")
                n += 1
            else:
                failures.append(f"{label}@{t:.3f}: {(run.stderr or '')[-160:]}")
        if failures:
            raise RuntimeError(
                f"S3 contact-sheet extraction lost {len(failures)}/{len(items)} frames: "
                + " | ".join(failures[:3])
            )
        if not n:
            return None
        rows = (n + cols - 1) // cols
        run = sh(["ffmpeg", "-v", "error", "-i", f"{tmp}/t%03d.png",
                  "-vf", f"tile={cols}x{rows}:padding=4:margin=4:color=0x1a1a1a",
                  "-frames:v", "1", "-y", out_png])
        if run.returncode != 0 or not os.path.isfile(out_png) or os.path.getsize(out_png) == 0:
            raise RuntimeError(f"S3 contact-sheet assembly failed: {(run.stderr or '')[-300:]}")
        return out_png
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build(task):
    from grading_contract import validate_task
    validate_task(task, "s_3")
    import numpy as np
    man = json.load(open(f"{task}/initial_state/manifest.json"))
    sub_p = f"{task}/output/cuts.json"
    if not os.path.exists(sub_p):
        return {"task": task, "error": "没有 output/cuts.json"}
    sub = json.load(open(sub_p))
    cuts = {c["clip"]: c for c in sub.get("cuts", [])}
    by = {c["id"]: c for c in man["clips"]}
    ev_dir = f"{task}/evidence"
    os.makedirs(ev_dir, exist_ok=True)

    beat = None
    tp = f"{task}/task_prompt.md"
    if os.path.exists(tp):
        import re
        m = re.search(r"covers:\s*\n+>\s*(.+?)\n", open(tp).read())
        beat = m.group(1).strip() if m else None

    # ---------------------------------------------------------------- 逐 clip
    per, order = [], [c["id"] for c in man["clips"]]
    prof_cache, words_cache = {}, {}
    for cid in order:
        c = by[cid]
        path = f"{task}/initial_state/{c['file']}"
        got = cuts.get(cid)
        rec = {"clip": cid, "clip_dur_s": c["clip_dur_s"],
               "given": bool(got)}
        if not got:
            per.append(rec)
            continue
        i_s, o_s = float(got.get("in_s", 0)), float(got.get("out_s", 0))
        rec.update({"in_s": i_s, "out_s": o_s,
                    "kept_s": round(o_s - i_s, 3),
                    "kept_pct": round(100 * (o_s - i_s) / max(1e-3, c["clip_dur_s"])),
                    "dropped_head_s": round(i_s, 3),
                    "dropped_tail_s": round(max(0.0, c["clip_dur_s"] - o_s), 3)})
        words = words_cache.setdefault(cid, words_of(path))
        prof = prof_cache.setdefault(cid, motion_profile(path))
        rec["speech_at_in"] = speech_at(words, i_s, "in")
        rec["speech_at_out"] = speech_at(words, o_s, "out")
        if prof:
            mi, mo = at_motion(prof, i_s), at_motion(prof, o_s)
            rec["motion"] = {
                "clip_median": prof["median"], "clip_p90": prof["p90"],
                "at_in": mi, "at_out": mo,
                # 相对这条 clip 自己的基线。>1 说明切在动作里，<1 说明切在静止处。
                "at_in_rel": round(mi / max(0.01, prof["median"]), 2) if mi is not None else None,
                "at_out_rel": round(mo / max(0.01, prof["median"]), 2) if mo is not None else None}
        # 被剪掉的台词：coverage 判分最要紧的一条证据
        drop = [w for w in words if w["e"] <= i_s + 0.05 or w["s"] >= o_s - 0.05]
        keep = [w for w in words if not (w["e"] <= i_s + 0.05 or w["s"] >= o_s - 0.05)]
        rec["kept_words"] = " ".join(w["w"] for w in keep)[:300]
        rec["dropped_words"] = " ".join(w["w"] for w in drop)[:300]
        per.append(rec)

    kept_tot = sum(r.get("kept_s", 0) for r in per)
    target = man.get("target_duration_s")

    # Descriptive duration statistics only, never an inference about effort/quality.
    given = [r for r in per if r.get("given")]
    shape = None
    if given:
        ks = [r["kept_s"] for r in given]
        inpct = [100 * r["in_s"] / max(1e-3, r["clip_dur_s"]) for r in given]
        mean_k = sum(ks) / len(ks)
        sd = (sum((x - mean_k) ** 2 for x in ks) / len(ks)) ** 0.5
        at_head = sum(1 for r in given if r["in_s"] < 0.02)
        shape = {
            "kept_s_stdev": round(sd, 3),
            "kept_s_cv": round(sd / max(1e-3, mean_k), 3),
            "in_pct_median": round(sorted(inpct)[len(inpct) // 2]),
            "clips_entered_at_head": at_head,
            "n_given": len(given),
            "interpretation_status": "descriptive_only",
            "note": "CV and head-entry counts do not prove editing quality or whether material was screened."}


    # ---------------------------------------------------------------- 逐接点
    joins = []
    # The actual assembled cut joins consecutive *submitted* clips.  Zipping
    # the full manifest silently skipped the new join created when a middle
    # clip was omitted.
    given_order = [cid for cid in order if cid in cuts]
    for a, b in zip(given_order, given_order[1:]):
        ra = next(r for r in per if r["clip"] == a)
        rb = next(r for r in per if r["clip"] == b)
        if not (ra.get("given") and rb.get("given")):
            continue
        pa = f"{task}/initial_state/{by[a]['file']}"
        pb = f"{task}/initial_state/{by[b]['file']}"
        # Seek inward when an out-point is at the file boundary; small duration
        # discrepancies can otherwise leave the join frame unavailable.
        fa = None
        for back in (0.04, 0.10, 0.20, 0.35):
            fa = gray(pa, ss=clamped_time(ra["out_s"] - back, ra["in_s"], ra["out_s"]),
                      bounds=(ra["in_s"], ra["out_s"]))
            if fa is not None:
                break
        fb = None
        for fwd in (0.0, 0.04, 0.10):
            fb = gray(pb, ss=clamped_time(rb["in_s"] + fwd, rb["in_s"], rb["out_s"]),
                      bounds=(rb["in_s"], rb["out_s"]))
            if fb is not None:
                break
        # 基线：两条 clip 中点的帧。同机位素材本来就像，那是素材不是剪辑。
        ma = gray(pa, ss=by[a]["clip_dur_s"] / 2)
        mb = gray(pb, ss=by[b]["clip_dur_s"] / 2)

        def sim(x, y):
            if x is None or y is None:
                return None
            x, y = x[0].ravel(), y[0].ravel()
            mae = float(np.abs(x - y).mean())
            xc, yc = x - x.mean(), y - y.mean()
            corr = float(np.dot(xc, yc) / (np.linalg.norm(xc) * np.linalg.norm(yc) + 1e-9))
            return {"mae": round(mae, 2), "corr": round(corr, 3)}

        joins.append({"join": len(joins) + 1, "leaving": a, "arriving": b,
                      "at_cut": sim(fa, fb),
                      "clip_baseline": sim(ma, mb),
                      "leaving_out_s": ra["out_s"], "arriving_in_s": rb["in_s"]})

    # ---------------------------------------------------------------- 联络表
    # Group navigational contact sheets by the evidence each rubric needs.
    sheets = {}
    P_ = lambda r: f"{task}/initial_state/{by[r['clip']]['file']}"
    # ① 留下的那一截长什么样：按时长采样，给 coverage 判。
    # 三帧对 0.4 秒的镜头够，对 1.5 秒的不够——brief 点名的动作可能就发生在没采到的
    # 那一段里。改成大约每 0.28 秒一帧，短镜头保底 3 帧，长镜头封顶 8 帧。
    kept = []
    for r in per:
        if not r.get("given"):
            continue
        a, b = r["in_s"], r["out_s"]
        span = b - a
        n = max(3, min(8, int(round(span / 0.28)) + 1))
        for i in range(n):
            frac = 0.02 + (0.94 * i / (n - 1)) if n > 1 else 0.5
            kept.append((P_(r), clamped_time(a + span * frac, a, b), f"{r['clip']}·{i + 1}/{n}", (a, b)))
    if sheet(kept, f"{ev_dir}/frames_kept.png", cols=6):
        sheets["kept_frames"] = {
            "file": "evidence/frames_kept.png",
            "note": "the kept span of each shot, sampled across its duration. A label "
                    "like `p03·2/5` is frame 2 of 5 from what p03 kept. This sheet "
                    "answers whether the pictures the brief describes are still there"}
    # ② 入点出点，各带一帧"被扔掉的那侧"：给 placement 判"切在哪儿"。
    # 只看留下的那一帧，判不出"该不该在这儿切"——得看见扔掉的是什么。
    # 入点前 0.3 秒和出点后 0.3 秒各一帧，四格一组。
    edges = []
    for r in per:
        if not r.get("given"):
            continue
        cd = by[r["clip"]]["clip_dur_s"]
        ins, outs = r["in_s"], r["out_s"]
        if ins > 0.12:
            edges.append((P_(r), max(0.0, ins - 0.30), f"{r['clip']} pre-in", (0., ins)))
        edges.append((P_(r), ins, f"{r['clip']} IN", (ins, outs)))
        edges.append((P_(r), clamped_time(outs - 0.08, ins, outs), f"{r['clip']} OUT", (ins, outs)))
        if outs + 0.30 < cd - 0.05:
            edges.append((P_(r), outs + 0.30, f"{r['clip']} post-out", (outs, cd)))
    if sheet(edges, f"{ev_dir}/frames_edges.png", cols=6):
        sheets["edge_frames"] = {
            "file": "evidence/frames_edges.png",
            "note": "four frames per shot: pre-in | IN | OUT | post-out. `pre-in` and "
                    "`post-out` are the material that was thrown away, and without them "
                    "there is no way to judge whether the cut belongs where it was put — "
                    "the kept frames alone cannot answer that"}
    # ③ 接点两侧：给 jumpcut 判"这一刀跳不跳"
    pairs = []
    for j in joins:
        ra, rb = next((x for x in per if x["clip"] == j["leaving"]), None), \
                 next((x for x in per if x["clip"] == j["arriving"]), None)
        if not (ra and rb and ra.get("given") and rb.get("given")):
            continue
        # 每侧两帧。一帧只说"这两格像不像"，两帧还能说出运动方向——同机位但主体
        # 在动，那是接得住的切；两侧都静止且构图一致，才是跳切。
        pairs.append((P_(ra), clamped_time(ra["out_s"] - .28, ra["in_s"], ra["out_s"]), f"j{j['join']}前-2", (ra["in_s"], ra["out_s"])))
        pairs.append((P_(ra), clamped_time(ra["out_s"] - .08, ra["in_s"], ra["out_s"]), f"j{j['join']}前-1", (ra["in_s"], ra["out_s"])))
        pairs.append((P_(rb), rb["in_s"], f"j{j['join']}后+1", (rb["in_s"], rb["out_s"])))
        pairs.append((P_(rb), clamped_time(rb["in_s"] + .20, rb["in_s"], rb["out_s"]), f"j{j['join']}后+2", (rb["in_s"], rb["out_s"])))
    if sheet(pairs, f"{ev_dir}/frames_joins.png", cols=4, w=300):
        sheets["join_frames"] = {
            "file": "evidence/frames_joins.png",
            "note": "the frames either side of each cut, laid out in pairs. `j05 pre` "
                    "sits next to `j05 post` — how alike those two look is whether the "
                    "cut reads as a jump"}

    # An empty model submission is a valid negative example, not grader
    # infrastructure failure.  Only require images when there is submitted
    # picture content for those images to show.
    required_sheets = ["kept_frames", "edge_frames"] if given_order else []
    if joins:
        required_sheets.append("join_frames")
    missing_sheets = [name for name in required_sheets if name not in sheets]
    if missing_sheets:
        raise RuntimeError(
            "S3 evidence is incomplete; missing required contact sheets: "
            + ", ".join(missing_sheets)
        )

    ev = {"task": task.split("training_task/")[-1],
          "beat": beat,
          "task_prompt": open(tp).read(),
          "continuous_video": s3_videos(task, man, sub),
          "target_duration_s": target,
          "assembled_duration_s": round(kept_tot, 3),
          "duration_delta_s": round(kept_tot - target, 3) if target else None,
          "n_clips": len(order), "n_given": sum(1 for r in per if r.get("given")),
          "trim_shape": shape,
          "clips": per, "joins": joins, "contact_sheets": sheets,
          "measurement_health": _HEALTH | {
              "required_contact_sheets": required_sheets,
              "contact_sheets_status": "ok",
          },
          "how_measured": {
              "speech": (f"faster-whisper {A.asr}, word-level timestamps, VAD on, "
                         f"condition_on_previous_text=false, contract={ASR_CONTRACT}"),
              "motion": "mean absolute frame-to-frame difference on 48x27 greyscale at 12fps; at_*_rel divides that by the clip's own median",
              "similarity": "single 48x27 greyscale frames; mae is mean absolute difference (0-255), corr is the mean-removed correlation",
              "trim_shape": "standard deviation and coefficient of variation of "
                            "kept_s, the median in point as a percentage of the clip, "
                            "and how many clips were cut from 0. Descriptive only; "
                            "neither proves or disproves quality.",
              "speech_is_not_grounds_for_deduction":
                  "Picture cuts may fall within speech; S3 sets picture points, "
                  "while sound is laid in S4.",
              "baseline": "clip_baseline is computed from the midpoint frame of each clip, and stands for how alike the material itself is"}}
    json.dump(ev, open(f"{ev_dir}/s3_evidence.json", "w"), ensure_ascii=False, indent=1)
    return ev


tasks = json.load(open(A.tasks)) if A.tasks else [A.task]
for t in tasks:
    if (not A.force and os.path.exists(f"{t}/evidence/s3_evidence.json")
            and json.load(open(f"{t}/evidence/s3_evidence.json")).get("continuous_video", {}).get("contract") == VIDEO_CONTRACT):
        print(f"  {t.split('training_task/')[-1]:<28} 已有，跳过", flush=True)
        continue
    e = build(t)
    if e.get("error"):
        print(f"  {t.split('training_task/')[-1]:<28} ★ {e['error']}", flush=True)
        continue
    print(f"  {e['task']:<28} {e['n_given']}/{e['n_clips']} 段  "
          f"{e['assembled_duration_s']}s / 目标 {e['target_duration_s']}s  "
          f"{len(e['joins'])} 接点", flush=True)
