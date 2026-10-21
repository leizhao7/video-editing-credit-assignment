"""Public construction draft: p2_refine.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/p2_refine.py [options]\n\n  --film\n  --source\n  --golden\n  --crop\n  --crop-expr\n  --afps\n  --wm\n  --fps\n  --win\n  --peak-band FLOAT  Diagnostic cosine-score band (default .02), not calibrated confidence.\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    try:
        from .temporal_alignment import (SCORE_KIND, paired_frame_scores,
                                         aligned_source_start, peak_interval)
    except ImportError:
        from temporal_alignment import (SCORE_KIND, paired_frame_scores,
                                        aligned_source_start, peak_interval)

    import argparse, csv, math, subprocess, time
    import numpy as np

    # Keep artifact-generating work inside the explicit entry point; importing must not run it.


    P = argparse.ArgumentParser()
    P.add_argument("--film", required=True)
    P.add_argument("--source", required=True)
    P.add_argument("--golden", required=True)
    P.add_argument("--crop", type=float, default=0.0)
    P.add_argument("--crop-expr", default="")
    P.add_argument("--afps", type=float, default=0.0)
    P.add_argument("--wm", default="")
    P.add_argument("--fps", type=float, default=24000/1001)
    P.add_argument("--win", type=float, default=1.5)     # search ± this many seconds
    P.add_argument("--peak-band", type=float, default=0.02,
                   help="diagnostic paired-cosine band around the peak, not a calibrated confidence interval")
    A = P.parse_args()
    if not math.isfinite(A.fps) or A.fps <= 0:
        P.error("--fps must be finite and positive")
    if not math.isfinite(A.win) or A.win < 0:
        P.error("--win must be finite and nonnegative")
    if not math.isfinite(A.peak_band) or A.peak_band <= 0:
        P.error("--peak-band must be finite and positive")

    W, H = 128, 58
    CROP = (A.crop_expr + ",") if A.crop_expr else (
        f"crop=iw:iw/{A.crop}:0:(ih-iw/{A.crop})/2," if A.crop else "")
    E = f"{DATA_ROOT}/exports/{A.film}"

    if A.wm:
        x0, x1, y0, y1 = [float(v) for v in A.wm.split(",")]
        MX = np.ones((H, W-1), bool); MX[int(y0*H):int(y1*H), int(x0*W):int(x1*W)-1] = False
        MY = np.ones((H-1, W), bool); MY[int(y0*H):int(y1*H)-1, int(x0*W):int(x1*W)] = False
    else:
        MX = np.ones((H, W-1), bool); MY = np.ones((H-1, W), bool)


    def win(path, t, dur, crop):
        vf = (CROP if crop else "") + f"fps={A.fps},scale={W}:{H},format=gray"
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0,t):.4f}", "-t", f"{dur:.4f}",
                            "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                           capture_output=True)
        if r.returncode:
            raise RuntimeError(f"alignment decoding failed: {r.stderr.decode('utf8', 'replace')[-400:]}")
        n = len(r.stdout)//(W*H)
        return np.frombuffer(r.stdout[:n*W*H], np.uint8).reshape(n, H, W).astype(np.float32)


    def desc(f):
        if len(f) == 0: return np.zeros((0, MX.sum()+MY.sum()), np.float32)
        gx, gy = np.diff(f, axis=2), np.diff(f, axis=1)
        d = np.concatenate([gx[:, MX], gy[:, MY]], 1)
        d -= d.mean(1, keepdims=True)
        return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)


    with open(f"{E}/p2_final.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        input_fields = reader.fieldnames or []
    t0 = time.time()
    out = []
    print(f"{'id':>6} {'dur':>6} {'旧入点':>10} {'精修后':>10} {'移动':>8} {'峰值':>6} "
          f"{'峰区宽':>6}  诊断（非置信度）")
    moved, plats = [], []
    empty = {"refined_src_s": "", "plateau_frames": "", "peak": "",
             "alignment_score_kind": SCORE_KIND, "alignment_fps": A.fps,
             "plateau_kind": "contiguous-peak-band-width-v1",
             "peak_band": A.peak_band, "near_peak_count": "", "selected_offset": "",
             "peak_left_offset": "", "peak_right_offset": "", "disjoint_near_peaks": "",
             "alignment_review_reason": ""}
    for r in rows:
        if not r["verdict"].startswith("MATCH"):
            out.append({**r, **empty}); continue
        g0, g1 = float(r["g_start_s"]), float(r["g_end_s"])
        st = float(r["matched_src_s"]); dur = g1 - g0
        lo = round(max(0.0, g0 + dur*0.15), 4)
        span = min(dur*0.7, 2.0)
        Q = desc(win(A.golden, lo, span, True))
        if len(Q) < 3:
            lo = round(max(0.0, g0), 4)
            Q = desc(win(A.golden, lo, dur, True))
        source_start = round(max(0.0, st - A.win), 4)
        S = desc(win(A.source, source_start, dur + 2*A.win, False))
        if len(Q) < 2 or len(S) < len(Q):
            out.append({**r, **empty, "verdict": "REVIEW(alignment)",
                        "alignment_review_reason": "insufficient_decoded_frames"}); continue
        acc = paired_frame_scores(S, Q)
        interval = peak_interval(acc, A.peak_band)
        b = interval["selected_offset"]; pk = float(acc[b])
        plateau = interval["plateau_frames"]
        ref = aligned_source_start(source_start, b, A.fps, lo, g0)
        dm = (ref - st)*A.fps
        v = "存在分离近峰" if interval["disjoint_near_peaks"] else "连续峰区"
        moved.append(abs(dm)); plats.append(plateau)
        print(f"{r['golden_id']:>6} {dur:5.2f}s {st:9.3f} {ref:10.3f} {dm:+7.1f}帧 "
              f"{pk:6.3f} {plateau:5d}帧  {v}")
        out.append({**r, **empty, **interval, "refined_src_s": round(ref, 4),
                    "peak": round(pk, 6),
                    "alignment_review_reason": "disjoint_near_peaks" if interval["disjoint_near_peaks"] else ""})

    moved, plats = np.array(moved), np.array(plats)
    if len(moved) == 0:
        # Zero refinable shots is not a crash, it is a finding: p2 matched nothing.
        # Write the pass-through file so the stage's output still exists and the failure
        # is visible in the match rate rather than in a traceback here.
        print("\n没有可精修的镜头 —— p2 一个都没匹配上，检查画框/裁切是否对齐")
        import sys as _s
        with open(f"{E}/p2_refined.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]) if out else list(dict.fromkeys(input_fields + list(empty)))); w.writeheader(); w.writerows(out)
        print(f"-> {E}/p2_refined.csv"); _s.exit(0)
    print(f"\n精修 {len(moved)} 条   移动 中位 {np.median(moved):.1f} 帧  最大 {moved.max():.0f} 帧")
    print(f"连续近峰宽度中位 {np.median(plats):.0f} 个采样帧（{A.fps:g} fps）；"
          f"分数带 {A.peak_band:g}，仅为启发式诊断，不代表对齐置信度")
    o = f"{E}/p2_refined.csv"
    with open(o, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"-> {o}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
