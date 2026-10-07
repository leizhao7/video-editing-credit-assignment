"""Public construction draft: build_cutlist.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/build_cutlist.py [options]\n\n  --film\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, json, os


    # Keep artifact-generating work inside the explicit entry point; importing must not run it.


    ROOT = DATA_ROOT
    P = argparse.ArgumentParser(); P.add_argument("--film", required=True)
    A = P.parse_args()
    T = f"{ROOT}/edit/task_{A.film}"
    u = json.load(open(f"{T}/s3_units.json"))
    units, fps = u["units"], u.get("fps", 24000 / 1001)

    # Accumulate integer frames for positions in the assembled S3 timeline.
    # Use the original reference manifest for timeline_s because image/audio inspection reads that reference.
    # These time bases may differ after per-unit frame rounding.


    import csv as _csv
    _gm = {r["id"]: r for r in _csv.DictReader(
        open(f"{ROOT}/exports/{A.film}/golden_clips/manifest.csv"))}

    cuts, fr = [], 0
    for k, (a, b) in enumerate(zip(units, units[1:])):
        fr += int(round(a["dur"] * fps))
        g = _gm.get(b["id"])
        t = float(g["start_s"]) if g else fr / fps
        cuts.append({"idx": k + 1, "after": a["id"], "before": b["id"],
                     "timeline_s": round(t, 4), "timeline_frame": fr,
                     "assembled_s": round(fr / fps, 4),
                     "after_kind": a["kind"], "before_kind": b["kind"]})

    os.makedirs(f"{T}/stages", exist_ok=True)
    json.dump({"stage": "cut_list", "film": A.film, "fps": fps,
               "n_cuts": len(cuts), "cuts": cuts},
              open(f"{T}/stages/cut_list.json", "w"), ensure_ascii=False, indent=1)
    from collections import Counter
    k = Counter((c["after_kind"], c["before_kind"]) for c in cuts)
    print(f"{A.film}: {len(units)} 单元 -> {len(cuts)} 个切点   总长 "
          f"{sum(x['dur'] for x in units):.2f}s")
    for (a, b), n in k.most_common(6):
        print(f"  {a:<12}→ {b:<12} {n}")
    print(f"-> {T}/stages/cut_list.json")


if __name__ == "__main__":
    main()
