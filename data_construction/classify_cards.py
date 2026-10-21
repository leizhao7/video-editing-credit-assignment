"""Public construction draft: classify_cards.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/classify_cards.py [options]\n\n  --film\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)
except ImportError:  # Direct script invocation.
    from release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)

import argparse, csv, json, subprocess
import numpy as np

ROOT = DATA_ROOT


def feats(path, t0, dur, crop, W=320, H=180):
    """Measure edge density/hardness for card classification.
    Type over a photographic background remains a known limitation.
    Misclassified cards can become golden-only units; inspect classifications.
    """
    vf = ((crop + ",") if isinstance(crop, str) and crop else
          (f"crop=iw:iw/{crop}:0:(ih-iw/{crop})/2," if crop else "")) + f"fps=6,scale={W}:{H}"
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
                        "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True)
    n = len(r.stdout) // (W * H * 3)
    if n < 1: return None
    f = np.frombuffer(r.stdout[:n * W * H * 3], np.uint8).reshape(n, H, W, 3).astype(np.float32)
    es, sh, hd = [], [], []
    for k in range(n):
        lum = f[k].mean(2)
        gx = np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))
        gy = np.abs(np.diff(lum, axis=0, append=lum[-1:, :]))
        loc = gx + gy
        e = loc > 25
        es.append(float(e.mean()))
        sh.append(float((loc[e] > 60).mean()) if e.any() else 0.0)
        hd.append(float((loc > 60).mean()))     # Hard-edge density (absolute, not diluted by the background).
    mid = f[n // 2]
    q = (mid // 32).astype(int)
    idx = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    # how many distinct text states: the lines animate in, so one grab loses steps
    states = 1
    if n > 2:
        d = [float(np.abs(f[k + 1] - f[k]).mean()) for k in range(n - 1)]
        states = 1 + sum(1 for x in d if x > max(3.0, 2.5 * np.median(d)))
    bg = [int(v) for v in mid.reshape(-1, 3)[
        np.bincount(idx.ravel()).argmax() == idx.ravel()].mean(0)] \
        if (idx.ravel() == np.bincount(idx.ravel()).argmax()).any() else None
    return {"edge": round(float(np.median(es)), 4),
            "sharp": round(float(np.median(sh)), 3),
            "hard": round(float(np.median(hd)), 4),
            "lum": round(float(mid.mean()), 1),
            "dom": round(float(np.bincount(idx.ravel()).max() / idx.size), 3),
            "states": int(min(states, 6)), "bg": bg}


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--film", required=True)
    A = P.parse_args()
    E = f"{ROOT}/exports/{A.film}"
    pb = json.load(open(f"{E}/probe.json"))
    G = f"{pb['dir']}/{pb['golden']}"
    gm = {r["id"]: r for r in csv.DictReader(open(f"{E}/golden_clips/manifest.csv"))}
    rows = list(csv.DictReader(open(f"{E}/p2_refined.csv")))

    out, n_card, n_live = {}, 0, 0
    for r in rows:
        # Colour-only candidates still need card/live filtering while awaiting
        # source-match review; they are not accepted source footage.
        if r["verdict"] not in {"NOT-IN-SOURCE", "REVIEW(colour)", "REVIEW(alignment)"}: continue
        g = gm[r["golden_id"]]
        f = feats(G, float(g["start_s"]), float(g["dur_s"]),
                  pb.get("crop_expr") or pb["crop"])
        if not f:
            out[r["golden_id"]] = {"kind": "live", "why": "no frames"}; n_live += 1; continue
        # hard edges, and enough of them
        kind = "card" if (f["edge"] >= 0.015 and f["sharp"] >= 0.35) else "live"
        out[r["golden_id"]] = {"kind": kind, **f}
        n_card += kind == "card"; n_live += kind == "live"

    json.dump(out, open(f"{E}/card_class.json", "w"), ensure_ascii=False, indent=1)
    print(f"{A.film}: {len(out)} unmatched shots -> title cards {n_card} / live action {n_live}")
    for k, v in out.items():
        print(f"  {k}  {v['kind']:<5} edge {v.get('edge','-'):<7} sharp {v.get('sharp','-'):<6} "
              f"lum {v.get('lum','-')}")


if __name__ == "__main__":
    main()
