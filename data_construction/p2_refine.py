"""Public construction draft: p2_refine.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/p2_refine.py [options]\n\n  --film\n  --source\n  --golden\n  --crop\n  --crop-expr\n  --afps\n  --wm\n  --fps\n  --win\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, csv, subprocess, time
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
    A = P.parse_args()

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
        n = len(r.stdout)//(W*H)
        return np.frombuffer(r.stdout[:n*W*H], np.uint8).reshape(n, H, W).astype(np.float32)


    def desc(f):
        if len(f) == 0: return np.zeros((0, MX.sum()+MY.sum()), np.float32)
        gx, gy = np.diff(f, axis=2), np.diff(f, axis=1)
        d = np.concatenate([gx[:, MX], gy[:, MY]], 1)
        d -= d.mean(1, keepdims=True)
        return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)


    rows = list(csv.DictReader(open(f"{E}/p2_final.csv")))
    t0 = time.time()
    out = []
    print(f"{'id':>6} {'dur':>6} {'旧入点':>10} {'精修后':>10} {'移动':>8} {'峰值':>6} "
          f"{'平台':>6}  判定")
    moved, plats = [], []
    for r in rows:
        if not r["verdict"].startswith("MATCH"):
            out.append({**r, "refined_src_s": "", "plateau_frames": "", "peak": ""}); continue
        g0, g1 = float(r["g_start_s"]), float(r["g_end_s"])
        st = float(r["matched_src_s"]); dur = g1 - g0
        lo = g0 + dur*0.15
        span = min(dur*0.7, 2.0)
        Q = desc(win(A.golden, lo, span, True))
        if len(Q) < 3:
            Q = desc(win(A.golden, g0, dur, True))
        S = desc(win(A.source, st - A.win, dur + 2*A.win, False))
        if len(Q) < 2 or len(S) < len(Q):
            out.append({**r, "refined_src_s": "", "plateau_frames": "", "peak": ""}); continue
        k = len(Q)
        C = S @ Q.T
        acc = np.array([C[o:o+k, np.arange(k)].mean() for o in range(len(S)-k+1)])
        b = int(np.argmax(acc)); pk = float(acc[b])
        plateau = int((acc > pk - 0.02).sum())          # 峰值附近多平
        ref = max(0.0, st - A.win + (b/A.fps) - (lo - g0))
        dm = (ref - st)*A.fps
        v = "OK" if plateau <= 6 else ("宽平台" if plateau <= 30 else "不可判")
        moved.append(abs(dm)); plats.append(plateau)
        print(f"{r['golden_id']:>6} {dur:5.2f}s {st:9.3f} {ref:10.3f} {dm:+7.1f}帧 "
              f"{pk:6.3f} {plateau:5d}帧  {v}")
        out.append({**r, "refined_src_s": round(ref, 4), "plateau_frames": plateau,
                    "peak": round(pk, 3)})

    moved, plats = np.array(moved), np.array(plats)
    if len(moved) == 0:
        # Zero refinable shots is not a crash, it is a finding: p2 matched nothing.
        # Write the pass-through file so the stage's output still exists and the failure
        # is visible in the match rate rather than in a traceback here.
        print("\n没有可精修的镜头 —— p2 一个都没匹配上，检查画框/裁切是否对齐")
        import sys as _s
        with open(f"{E}/p2_refined.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        print(f"-> {E}/p2_refined.csv"); _s.exit(0)
    print(f"\n精修 {len(moved)} 条   移动 中位 {np.median(moved):.1f} 帧  最大 {moved.max():.0f} 帧")
    print(f"对齐平台宽度  中位 {np.median(plats):.0f} 帧   "
          f"<=6帧(可判) {int((plats<=6).sum())}  7-30帧 {int(((plats>6)&(plats<=30)).sum())}  "
          f">30帧(不可判) {int((plats>30).sum())}")
    o = f"{E}/p2_refined.csv"
    with open(o, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"-> {o}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
