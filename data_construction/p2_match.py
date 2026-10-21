"""Public construction draft: p2_match.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/p2_match.py [options]\n\n  --film\n  --source\n  --golden\n  --crop\n  --crop-expr\n  --afps\n  --wm\n  --out\n  --paired-colour-threshold FLOAT  Explicit paired-score acceptance policy; default: review colour candidates.\n  -h, --help  Show this static help without loading optional dependencies.\n')
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
                                         aligned_source_start, colour_verdict)
    except ImportError:
        from temporal_alignment import (SCORE_KIND, paired_frame_scores,
                                        aligned_source_start, colour_verdict)

    import argparse, csv, json, os, subprocess, sys, time
    import numpy as np, cv2


    # Keep artifact-generating work inside the explicit entry point; importing must not run it.


    P = argparse.ArgumentParser()
    P.add_argument("--film", required=True)
    P.add_argument("--source", required=True)          # proxy is fine, same timebase
    P.add_argument("--golden", required=True)
    P.add_argument("--crop", type=float, default=0.0)
    P.add_argument("--crop-expr", default="")
    P.add_argument("--afps", type=float, default=0.0)  # golden letterbox ratio
    P.add_argument("--wm", default="")                 # "x0,x1,y0,y1" normalised, golden overlay
    P.add_argument("--out", required=True)
    P.add_argument("--paired-colour-threshold", type=float, default=None,
                   help="explicit mean paired-cosine threshold; colour-only candidates require review by default")
    A = P.parse_args()
    try:
        colour_verdict(0.0, A.paired_colour_threshold)
    except ValueError as exc:
        P.error(str(exc))

    W, H = 96, 44
    # Temporal consistency requires multiple samples within a shot.
    # The caller derives the sampling rate from the input shot lengths.


    FPS = A.afps or 4.0
    CROP = (A.crop_expr + ",") if A.crop_expr else (
        f"crop=iw:iw/{A.crop}:0:(ih-iw/{A.crop})/2," if A.crop else "")
    E = f"{DATA_ROOT}/exports/{A.film}"
    t0 = time.time()


    def dec(path, vf, w=W, h=H):
        """Raise on failed or empty decoding so invalid frame arrays cannot enter the cache."""
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf,
                            "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
        n = len(r.stdout) // (w * h)
        if r.returncode or n == 0:
            raise RuntimeError(f"Decoding failed: {os.path.basename(path)}  rc={r.returncode}  "
                               f"{n} frames\n{r.stderr.decode('utf8','replace')[-600:]}")
        return np.frombuffer(r.stdout[:n * w * h], np.uint8).reshape(n, h, w).astype(np.float32)


    # ---- watermark mask, applied identically to both sides ----
    if A.wm:
        x0, x1, y0, y1 = [float(v) for v in A.wm.split(",")]
        MX = np.ones((H, W - 1), bool); MX[int(y0*H):int(y1*H), int(x0*W):int(x1*W)-1] = False
        MY = np.ones((H - 1, W), bool); MY[int(y0*H):int(y1*H)-1, int(x0*W):int(x1*W)] = False
        print(f"Watermark excluded at x{x0}-{x1} y{y0}-{y1}; descriptor dimensions reduced by "
              f"{100*(1-(MX.sum()+MY.sum())/(MX.size+MY.size)):.1f}%")
    else:
        MX = np.ones((H, W - 1), bool); MY = np.ones((H - 1, W), bool)


    def desc(f):
        if len(f) == 0: return np.zeros((0, MX.sum() + MY.sum()), np.float32)
        gx, gy = np.diff(f, axis=2), np.diff(f, axis=1)
        d = np.concatenate([gx[:, MX], gy[:, MY]], 1)
        d -= d.mean(1, keepdims=True)
        return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)


    # The cache is keyed by sampling rate as well as size — a run at a different --afps
    # must not silently reuse frames sampled at another rate.
    cache = f"{E}/src_gray{W}_{FPS:g}.npy"
    src = None
    if os.path.exists(cache):
        src = np.load(cache).astype(np.float32)
        if len(src) == 0:
            os.remove(cache); src = None          # never trust an empty cache
        else:
            print(f"Reusing cache {src.shape}")
    if src is None:
        src = dec(A.source, f"fps={FPS},scale={W}:{H},format=gray")
        np.save(cache, src.astype(np.uint8)); print(f"source {len(src)} frames  {time.time()-t0:.0f}s")
    gold = dec(A.golden, f"{CROP}fps={FPS},scale={W}:{H},format=gray")
    SD, GD = desc(src), desc(gold)
    glum = gold.mean((1, 2)); BLACK = glum.max() * 0.08
    print(f"golden {len(gold)} frames   descriptor dimensions {SD.shape[1]}   {time.time()-t0:.0f}s")

    gm = list(csv.DictReader(open(f"{E}/golden_clips/manifest.csv")))
    cm = list(csv.DictReader(open(f"{E}/source_manifest.csv")))
    cstart = np.array([float(r["start_s"]) for r in cm])

    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    GW, GH = 960, int(960 / (A.crop or 2.39))
    WMASK = np.full((GH, GW), 255, np.uint8)
    if A.wm:
        WMASK[int(y0*GH):int(y1*GH), int(x0*GW):int(x1*GW)] = 0


    def grab(path, t, crop):
        vf = (CROP if crop else "") + f"scale={GW}:{GH}"
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path,
                            "-frames:v", "1", "-vf", vf, "-f", "rawvideo",
                            "-pix_fmt", "gray", "-"], capture_output=True)
        return np.frombuffer(r.stdout[:GW*GH], np.uint8).reshape(GH, GW) \
            if len(r.stdout) >= GW*GH else None


    def inl(ga, sa):
        if ga is None or sa is None: return 0
        k1, d1 = orb.detectAndCompute(ga, WMASK)        # golden: skip the overlay
        k2, d2 = orb.detectAndCompute(sa, None)
        if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12: return 0
        m = bf.match(d1, d2)
        if len(m) < 12: return 0
        p1 = np.float32([k1[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
        p2 = np.float32([k2[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
        return int(mask.sum()) if mask is not None else 0


    CW, CH, RF = 64, 28, 12.0


    def cwin(path, t, dur, crop):
        vf = (CROP if crop else "") + f"fps={RF},scale={CW}:{CH}"
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0,t):.3f}", "-t", f"{dur:.3f}",
                            "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True)
        if r.returncode:
            raise RuntimeError(f"colour alignment decoding failed: {r.stderr.decode('utf8', 'replace')[-400:]}")
        n = CW*CH*3; k = len(r.stdout)//n
        if not k: return np.zeros((0, CH*CW*3), np.float32)
        f = np.frombuffer(r.stdout[:k*n], np.uint8).reshape(k, CH, CW, 3).astype(np.float32)
        v = []
        for c in range(3):
            x = f[:, :, :, c]
            v.append(((x - x.mean((1, 2), keepdims=True)) /
                      (x.std((1, 2), keepdims=True) + 1e-6)).reshape(k, -1))
        v = np.concatenate(v, 1)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


    rows = []
    print(f"\n{'id':>6} {'span':>14} {'dur':>6} {'clip':>7} {'src_tc':>10} {'ORB':>5} {'col':>6}  verdict")
    for r in gm:
        # Candidate fields stay separate from accepted source fields. In particular,
        # REVIEW(colour) must not acquire a source in-point in downstream stages.
        extra = {"colour_score_kind": "", "colour_threshold": "",
                 "candidate_source_clip": "", "candidate_src_s": "", "review_reason": ""}
        gid, t0g, t1g = r["id"], float(r["start_s"]), float(r["end_s"])
        lo, hi = t0g + (t1g-t0g)*.15, t1g - (t1g-t0g)*.15
        idx = [i for i in range(int(lo*FPS), int(hi*FPS)+1)
               if 0 <= i < len(gold) and glum[i] > BLACK]
        if len(idx) < 2:                     # For short shots, use the full span instead of only the middle 70%.
            idx = [i for i in range(int(t0g*FPS), int(t1g*FPS)+1)
                   if 0 <= i < len(gold) and glum[i] > BLACK]
        span = f"{t0g:6.2f}-{t1g:6.2f}"
        if len(idx) < 2:
            # Insufficient samples do not imply a black shot. Check denser samples from this shot.
            rr = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0g:.3f}",
                                 "-t", f"{t1g-t0g:.3f}", "-i", A.golden, "-vf",
                                 CROP + "fps=12,scale=64:26,format=gray", "-f", "rawvideo",
                                 "-pix_fmt", "gray", "-"], capture_output=True)
            k = len(rr.stdout)//(64*26)
            pk = (np.frombuffer(rr.stdout[:k*64*26], np.uint8).reshape(k, 26, 64)
                  .astype(np.float32).mean((1, 2)).max()) if k else 0.0
            if pk >= 14:                     # Image content is dark or brief; use every frame in the span.
                idx = [i for i in range(int(t0g*FPS), int(t1g*FPS)+1) if 0 <= i < len(gold)]
        if len(idx) < 2:
            print(f"{gid:>6} {span:>14} {t1g-t0g:5.2f}s {'—':>7} {'—':>10} {'—':>5} {'—':>6}  BLACK")
            rows.append([gid, t0g, t1g, "", "", "", 0, 0.0, "BLACK", *extra.values()]); continue
        idx = idx[:8]; off = np.array(idx)-idx[0]
        acc = paired_frame_scores(SD, GD[idx], off)
        if not len(acc):
            rows.append([gid, t0g, t1g, "", "", "", 0, 0.0, "NOT-IN-SOURCE", *extra.values()]); continue
        top, m = [], acc.copy()
        for _ in range(3):
            i = int(np.argmax(m)); top.append(i/FPS); m[max(0, i-40):i+40] = -9
        mid = idx[len(idx)//2]; dt = (mid-idx[0])/FPS
        ga = grab(A.golden, mid/FPS, True)
        sc = [inl(ga, grab(A.source, t+dt, False)) for t in top]
        j = int(np.argmax(sc)); col = 0.0; v = None
        if sc[j] >= 25: st, v = top[j], "MATCH"
        else:
            query_start = round(max(0.0, lo), 3)
            gq = cwin(A.golden, query_start, min(1.5, hi-lo), True)
            if len(gq) >= 3:
                s2 = []
                for t in top:
                    source_start = round(max(0.0, t-1.0), 3)
                    scores = paired_frame_scores(cwin(A.source, source_start, 3.5, False), gq)
                    if len(scores):
                        b = int(np.argmax(scores))
                        candidate = aligned_source_start(source_start, b, RF, query_start, t0g)
                        s2.append((float(scores[b]), candidate))
                if s2:
                    col, st = max(s2)
                    v = colour_verdict(col, A.paired_colour_threshold)
                    ci = int(np.searchsorted(cstart, st, side="right")-1)
                    extra.update(colour_score_kind=SCORE_KIND,
                                 colour_threshold=A.paired_colour_threshold if A.paired_colour_threshold is not None else "",
                                 candidate_source_clip=cm[max(0, ci)]["id"],
                                 candidate_src_s=round(st, 4),
                                 review_reason="uncalibrated_colour_fallback" if v == "REVIEW(colour)" else "")
        if v and v.startswith("MATCH"):
            ci = int(np.searchsorted(cstart, st, side="right")-1); c = cm[max(0, ci)]
            print(f"{gid:>6} {span:>14} {t1g-t0g:5.2f}s {c['id']:>7} "
                  f"{int(st//60):3d}:{st%60:05.2f} {max(sc[j],0):5d} {col:6.3f}  {v}")
            rows.append([gid, t0g, t1g, c["id"], c["start_s"], round(st, 4), max(sc[j], 0),
                         round(col, 6), v, *extra.values()])
        else:
            v = v or "NOT-IN-SOURCE"
            print(f"{gid:>6} {span:>14} {t1g-t0g:5.2f}s {'—':>7} {'—':>10} {sc[j]:5d} {col:6.3f}"
                  f"  {v}")
            rows.append([gid, t0g, t1g, "", "", "", sc[j], round(col, 6), v, *extra.values()])

    with open(A.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["golden_id", "g_start_s", "g_end_s", "source_clip", "clip_start_s",
                    "matched_src_s", "orb_inliers", "colour_score", "verdict",
                    "colour_score_kind", "colour_threshold", "candidate_source_clip",
                    "candidate_src_s", "review_reason"])
        w.writerows(rows)
    from collections import Counter
    mt = [r for r in rows if r[8].startswith("MATCH")]
    cl = [r[3] for r in mt]
    print(f"\n{dict(Counter(r[8] for r in rows))}   total {len(rows)}")
    print(f"Matched {len(mt)} shots to {len(set(cl))} clips; {len(mt)-len(set(cl))} clip reuses")
    print(f"-> {A.out}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
