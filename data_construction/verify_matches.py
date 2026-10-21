"""Public construction draft: verify_matches.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/verify_matches.py [options]\n\n  -j, --workers\n  --out\n  --all  Process all supplied inputs explicitly\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, csv, glob, json, os, subprocess, sys, tempfile
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    # Keep artifact-generating work inside the explicit entry point; importing must not run it.


    ROOT = DATA_ROOT
    P = argparse.ArgumentParser()
    P.add_argument("-j", "--workers", type=int, default=6)
    P.add_argument("--out", default=f"{ROOT}/match_confidence.json")
    P.add_argument("--all", action="store_true", help="Process all supplied inputs explicitly")
    A = P.parse_args()
    require_target(P, A)

    FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)
    FLAT = 3.0          # Frames below this standard deviation are treated as flat and excluded.


    def gray(src, t, crop):
        """one 64x36 luma frame, or None if it cannot be read"""
        with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as f:
            p = f.name
        try:
            vf = ((crop + ",") if crop else "") + "scale=64:36"
            r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}",
                                "-i", src, "-map", "0:v:0", "-vf", vf, "-frames:v", "1",
                                "-pix_fmt", "gray", "-y", p], capture_output=True)
            if r.returncode:
                return None
            d = open(p, "rb").read()
            try:
                i = d.index(b"255\n") + 4
            except ValueError:
                return None
            a = np.frombuffer(d[i:], dtype=np.uint8).astype(float)
            return a[:64 * 36].reshape(36, 64) if a.size >= 64 * 36 else None
        finally:
            if os.path.exists(p):
                os.remove(p)


    def corr(a, b):
        """Return None when either input is too flat for correlation.
        A zero denominator is missing measurement evidence, not a measured zero similarity."""
        a = a - a.mean()
        b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a * b).sum() / d) if d else None


    def score(job):
        slug, sid, gold, src, gcrop, scrop, g_start, dur, s_at = job
        best = -2.0
        for fr in FRACS:
            off = dur * fr
            f1 = gray(gold, g_start + off, gcrop)
            f2 = gray(src, s_at + off, scrop)
            if f1 is None or f2 is None:
                continue
            if f1.std() < FLAT or f2.std() < FLAT:
                continue                       # A flat frame on either side leaves no comparable content.
            c = corr(f1, f2)
            if c is not None:
                best = max(best, c)
        return f"{slug}/{sid}", (None if best < -1 else round(best, 4))


    jobs = []
    matches = sorted(glob.glob(f"{ROOT}/exports/*/p2_refined.csv"))
    if not matches:
        raise SystemExit("No p2_refined.csv found. Complete pipeline preprocessing, then run "
                         "python data_construction/run_script_queue.py --film <slug> --prep-only.")
    for f in matches:
        slug = f.split("/")[-2]
        inputs = f"{ROOT}/edit/task_{slug}/stages/script_inputs.json"
        if not os.path.isfile(inputs):
            raise SystemExit(f"Missing {inputs}. Run python data_construction/run_script_queue.py "
                             f"--film {slug} --prep-only before verify_matches.py.")
        pb = json.load(open(f"{ROOT}/exports/{slug}/probe.json"))
        inp = {s["id"]: s for s in json.load(open(inputs))["shots"]}
        gold, src = f"{pb['dir']}/{pb['golden']}", f"{pb['dir']}/{pb['source']}"
        gcrop, scrop = pb.get("crop_expr") or "", pb.get("src_crop_expr", "")
        for r in csv.DictReader(open(f)):
            v = r.get("refined_src_s")
            sh = inp.get(r["golden_id"])
            if v in (None, "", "None") or not sh:
                continue
            jobs.append((slug, r["golden_id"], gold, src, gcrop, scrop,
                         sh["start_s"], sh["dur_s"], float(v)))

    if not jobs:
        raise SystemExit("No eligible source matches to compare. Check p2_refined.csv and script_inputs.json; "
                         "no match-confidence evidence was produced.")

    print(f"  {len(jobs)} matched shots; {A.workers} workers", flush=True)
    out = {}
    with ThreadPoolExecutor(max_workers=A.workers) as ex:
        for n, (k, v) in enumerate(ex.map(score, jobs), 1):
            out[k] = v
            if n % 200 == 0:
                print(f"    {n}/{len(jobs)}", flush=True)
    json.dump(out, open(A.out, "w"), indent=0)

    vals = sorted(v for v in out.values() if v is not None)
    print(f"\n  Wrote {A.out}")
    if not vals:
        raise SystemExit("No measurable comparisons: frames could not be decoded or were too flat for correlation. "
                         f"Saved {len(out)} null measurements to {A.out}; no match-confidence evidence is available.")
    print(f"  Measured {len(vals)} shots: median {vals[len(vals)//2]:.3f}  "
          f"10% {vals[len(vals)//10]:.3f}  minimum {vals[0]:.3f}")
    for thr in (0.5, 0.6, 0.7):
        print(f"  Below {thr}: {sum(1 for x in vals if x < thr)} shots "
              f"({100*sum(1 for x in vals if x < thr)/len(vals):.1f}%)")
    print(f"  Unmeasurable (frame extraction failed or no non-flat frame pair): {sum(1 for v in out.values() if v is None)}")


if __name__ == "__main__":
    main()
