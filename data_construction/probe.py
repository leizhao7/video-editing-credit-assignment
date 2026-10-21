"""Public construction draft: probe.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/probe.py [options]\n\n  --dir\n  --slug\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)
except ImportError:  # Direct script invocation.
    from release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)

import argparse, json, os, re, subprocess, sys
import numpy as np

ROOT = DATA_ROOT


def spec(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
                        "-show_entries", "format=duration", "-of", "json", p],
                       capture_output=True, text=True).stdout
    d = json.loads(r)
    s = d["streams"][0]
    n, dn = s["r_frame_rate"].split("/")
    fps = int(n) / int(dn)
    dur = float(d["format"]["duration"])

    vdur = 0.0
    for k in ("duration", "nb_frames"):
        v = s.get(k)
        if v in (None, "N/A"): continue
        try: vdur = float(v) / (fps if k == "nb_frames" else 1.0)
        except ValueError: vdur = 0.0
        if vdur > 0: break
    if vdur <= 0:                       # mkv rarely carries either; count the packets
        q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-count_packets", "-show_entries", "stream=nb_read_packets",
                            "-of", "csv=p=0", p], capture_output=True, text=True).stdout
        try: vdur = float(q.strip().strip(",")) / fps
        except ValueError: vdur = dur
    return {"w": s["width"], "h": s["height"], "fps": fps, "dur": dur,
            "vdur": round(vdur, 3)}


def content_ratio(path, dur, n=20):
    """Measure content aspect ratio while ignoring letterbox bars.
    Raise if measurement is insufficient rather than assuming no bars."""
    rs = []
    for t in np.linspace(dur * 0.08, dur * 0.92, n):
        out = subprocess.run(["ffmpeg", "-v", "info", "-ss", f"{t:.2f}", "-i", path,
                              "-vf", "cropdetect=limit=0.09:round=2:reset=1",
                              "-frames:v", "12", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        m = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", out)
        if m:
            w, h, _, _ = map(int, m[-1])
            if w > 40 and h > 40: rs.append(w / h)
    if len(rs) < 8:
        raise RuntimeError(f"Cannot measure content aspect ratio: {len(rs)}/{n} samples returned results; "
                           f"cropdetect failed; this does not mean there are no letterbox bars: {os.path.basename(path)}")
    rs = np.array(rs)
    # the mode is more robust than the mean: a few dark frames give silly crops
    hist, edges = np.histogram(rs, bins=np.arange(1.2, 3.0, 0.02))
    k = int(np.argmax(hist))
    sel = rs[(rs >= edges[k]) & (rs < edges[k + 2])]
    r = float(np.median(sel)) if len(sel) else float(np.median(rs))
    conf = len(sel) / len(rs)
    if conf < 0.6:
        # genuinely variable — IMAX sections shift ratio mid-film. Not an error, but
        # the single number below is a compromise and downstream crops will be soft.
        print(f"  Warning: {os.path.basename(path)} has a variable aspect ratio; the modal band covers only {conf:.0%}"
              f" (range {rs.min():.2f}-{rs.max():.2f}); using {r:.4f}")
    return r


def watermark(path, dur, crop, W=320, H=180):
    """A persistent bug never goes dark. The naive tests both fail:
    temporal std is dominated by the background showing through the logo, and the
    temporal MINIMUM is 0 everywhere because every trailer dips to black at least
    once. The 12th percentile over time ignores those dips and leaves the bug
    standing. Returns normalised x0,x1,y0,y1 of the content area, or None."""
    vf = (f"crop=iw:iw/{crop}:0:(ih-iw/{crop})/2," if crop else "") + \
         f"fps=1,scale={W}:{H},format=gray"
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf,
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    n = len(r.stdout) // (W * H)
    if n < 20: return None
    f = np.frombuffer(r.stdout[:n * W * H], np.uint8).reshape(n, H, W).astype(np.float32)
    p12 = np.percentile(f, 12, axis=0)
    edge = np.ones((H, W), bool)
    edge[int(H * .16):int(H * .84), int(W * .13):int(W * .87)] = False   # bugs sit at edges
    thr = max(35.0, float(np.percentile(p12[edge], 99.2)))
    m = (p12 > thr) & edge
    if m.sum() < 6: return None
    ys, xs = np.where(m)
    box = [xs.min() / W, (xs.max() + 1) / W, ys.min() / H, (ys.max() + 1) / H]
    if (box[1] - box[0]) * (box[3] - box[2]) > 0.12: return None    # too big for a bug
    pad = 0.015
    return [round(max(0, box[0] - pad), 3), round(min(1, box[1] + pad), 3),
            round(max(0, box[2] - pad), 3), round(min(1, box[3] + pad), 3)]


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--dir", required=True)      # sources/<Film>
    P.add_argument("--slug", required=True)
    A = P.parse_args()

    files = os.listdir(A.dir)
    src = next((f for f in files if re.search(r"source", f, re.I)
                and f.lower().endswith((".mkv", ".mp4", ".mov"))), None)
    gold = next((f for f in files if re.search(r"trailer", f, re.I)
                 and f.lower().endswith((".mkv", ".mp4", ".mov"))), None)
    if not src or not gold:
        print(json.dumps({"error": "missing source or trailer", "files": files})); sys.exit(1)
    S, G = spec(f"{A.dir}/{src}"), spec(f"{A.dir}/{gold}")

    sr = content_ratio(f"{A.dir}/{src}", S["dur"])
    gr = content_ratio(f"{A.dir}/{gold}", G["dur"])
    frame = G["w"] / G["h"]
    # Crop letterboxing and pillarboxing according to the measured content aspect ratio.
    # Apply the appropriate crop independently to source and reference media before matching.


    sframe = S["w"] / S["h"]
    src_crop_expr = ""
    if abs(sr - sframe) > 0.04:
        src_crop_expr = (f"crop=iw:iw/{sr:.4f}:0:(ih-iw/{sr:.4f})/2" if sr > sframe
                         else f"crop=ih*{sr:.4f}:ih:(iw-ih*{sr:.4f})/2:0")

    crop, crop_expr, crop_mode = 0.0, "", "none"
    if abs(gr - frame) > 0.04:
        crop = round(gr, 4)
        if gr > frame:
            crop_expr = f"crop=iw:iw/{crop}:0:(ih-iw/{crop})/2"; crop_mode = "letterbox"
        else:
            crop_expr = f"crop=ih*{crop}:ih:(iw-ih*{crop})/2:0"; crop_mode = "pillarbox"
    wm = watermark(f"{A.dir}/{gold}", G["dur"], crop)

    out = {"slug": A.slug, "dir": os.path.abspath(A.dir), "source": src, "golden": gold,
           "src": S, "gold": G,
           "src_ratio": round(sr, 4), "gold_ratio": round(gr, 4),
           "crop": crop, "crop_expr": crop_expr, "crop_mode": crop_mode,
           "src_crop_expr": src_crop_expr, "wm": wm,
           "fps_match": abs(S["fps"] - G["fps"]) < 0.01,
           "analysis_fps": 4.0,
           "proxy_fps": round(S["fps"], 6)}
    os.makedirs(f"{ROOT}/exports/{A.slug}", exist_ok=True)
    json.dump(out, open(f"{ROOT}/exports/{A.slug}/probe.json", "w"), indent=1)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
