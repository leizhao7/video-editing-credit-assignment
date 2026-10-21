"""Public construction draft: picture_cont.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/picture_cont.py [options]\n\n  --film\n  --margin  Maximum similarity drop from the within-shot baseline for continuity\n  --floor  Minimum cross-boundary similarity for continuity\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, json, os, subprocess, sys
    import numpy as np

    # Keep artifact-generating work inside the explicit entry point; importing must not run it.


    ROOT = DATA_ROOT
    P = argparse.ArgumentParser()
    P.add_argument("--film", required=True)
    P.add_argument("--margin", type=float, default=0.25,
                   help="Maximum similarity drop from the within-shot baseline for continuity")
    P.add_argument("--floor", type=float, default=0.60,
                   help="Minimum cross-boundary similarity for continuity")
    A = P.parse_args()

    E, T = f"{ROOT}/exports/{A.film}", f"{ROOT}/edit/task_{A.film}"
    pb = json.load(open(f"{E}/probe.json"))
    G = f"{pb['dir']}/{pb['golden']}"
    CROP = (pb.get("crop_expr", "") + ",") if pb.get("crop_expr") else (
        f"crop=iw:iw/{pb['crop']}:0:(ih-iw/{pb['crop']})/2," if pb["crop"] else "")
    j = json.load(open(f"{T}/stages/cut_list.json"))
    cuts, FPS = j["cuts"], j["fps"]
    W, H = 64, 36


    def frame(t):
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.4f}", "-i", G,
                            "-frames:v", "1", "-vf", CROP + f"scale={W}:{H},format=gray",
                            "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
        return (np.frombuffer(r.stdout[:W * H], np.uint8).astype(np.float64)
                if len(r.stdout) >= W * H else None)


    FLAT = 1e-6


    def sim(a, b):
        """Return correlation and whether either input is too flat to correlate.
        Distinguish a flat-frame sentinel from an actual measured similarity."""
        a, b = a - a.mean(), b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < FLAT and nb < FLAT: return 1.0, "both"
        if na < FLAT or nb < FLAT: return 0.0, ("first" if na < FLAT else "second")
        return float(a @ b / (na * nb)), None


    rows, n_cont = [], 0
    for c in cuts:
        t = c["timeline_s"]
        x, y = frame(t - 5 / FPS), frame(t - 3 / FPS)          # Within one shot, two frames apart.
        a, b = frame(t - 1.5 / FPS), frame(t + 0.5 / FPS)      # Across the boundary, two frames apart.
        if any(z is None for z in (x, y, a, b)):
            rows.append({"idx": c["idx"], "continuous": None}); continue
        base, base_flat = sim(x, y)
        cross, cross_flat = sim(a, b)
        cont = cross >= base - A.margin and cross >= A.floor
        n_cont += cont
        r = {"idx": c["idx"], "continuous": bool(cont),
             "baseline": round(base, 3), "across": round(cross, 3)}
        if base_flat or cross_flat:
            r["flat"] = {"baseline": base_flat, "across": cross_flat}
            r["note"] = "At least one frame is flat (black/white). Correlation is undefined; the value is a sentinel, not a measurement."
        rows.append(r)

    out = f"{T}/stages/picture_measured.json"
    json.dump({"note": "Measured picture continuity across boundaries. no_change requires continuous=true.",
               "margin": A.margin, "n": len(rows), "n_continuous": n_cont, "cuts": rows},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{A.film}: {len(rows)} boundaries; picture continuous at {n_cont} -> {out}")


if __name__ == "__main__":
    main()
