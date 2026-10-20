"""Public construction draft: shot_luma.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/shot_luma.py [options]\n\n  -j, --workers\n  --out\n  --all  Process all supplied inputs explicitly\n  -h, --help  Show this static help without loading optional dependencies.\n')
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

    ROOT = DATA_ROOT
    sys.path.insert(0, CODE_ROOT)
    from build_s1_tasks import LUMA_SAMPLES, featureless   # 同一套判据，见 build_s1_tasks
    P = argparse.ArgumentParser()
    P.add_argument("-j", "--workers", type=int, default=8)
    P.add_argument("--out", default=f"{ROOT}/shot_luma.json")
    P.add_argument("--all", action="store_true", help="Process all supplied inputs explicitly")
    A = P.parse_args()
    require_target(P, A)


    def frame(src, t, crop):
        with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as f:
            p = f.name
        try:
            vf = ((crop + ",") if crop else "") + "scale=64:36"
            if subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}",
                               "-i", src, "-map", "0:v:0", "-vf", vf, "-frames:v", "1",
                               "-pix_fmt", "gray", "-y", p], capture_output=True).returncode:
                return None
            d = open(p, "rb").read()
            try:
                i = d.index(b"255\n") + 4
            except ValueError:
                return None
            a = np.frombuffer(d[i:], dtype=np.uint8).astype(float)
            return a[:64 * 36] if a.size >= 64 * 36 else None
        finally:
            if os.path.exists(p):
                os.remove(p)


    def measure(job):
        slug, sid, src, start, dur, crop = job
        best = None
        for fr in LUMA_SAMPLES:
            a = frame(src, start + dur * fr, crop)
            if a is None:
                continue
            m, s = float(a.mean()), float(a.std())
            if best is None or m > best[0]:
                best = (m, s)
        return f"{slug}/{sid}", (None if best is None else [round(best[0], 1), round(best[1], 1)])


    def main():
        jobs = []
        inputs = sorted(glob.glob(f"{ROOT}/edit/task_*/stages/script_inputs.json"))
        for probe in sorted(glob.glob(f"{ROOT}/exports/*/probe.json")):
            slug = os.path.basename(os.path.dirname(probe))
            required = f"{ROOT}/edit/task_{slug}/stages/script_inputs.json"
            if not os.path.isfile(required):
                raise SystemExit(f"Missing {required}. Run python data_construction/run_script_queue.py "
                                 f"--film {slug} --prep-only before shot_luma.py.")
        if not inputs:
            raise SystemExit("No script_inputs.json found. Complete preprocessing, then run "
                             "python data_construction/run_script_queue.py --film <slug> --prep-only.")
        for p in inputs:
            slug = p.split("/")[-3][5:]
            pb = json.load(open(f"{ROOT}/exports/{slug}/probe.json"))
            src = f"{pb['dir']}/{pb['golden']}"
            crop = pb.get("crop_expr") or ""
            for s in json.load(open(p))["shots"]:
                jobs.append((slug, s["id"], src, s["start_s"], s["dur_s"], crop))

        if not jobs:
            raise SystemExit("No shots to measure in script_inputs.json. Check preprocessing and rerun "
                             "python data_construction/run_script_queue.py --film <slug> --prep-only.")

        print(f"  {len(jobs)} 个预告片镜头，{A.workers} 路并发", flush=True)
        out = {}
        with ThreadPoolExecutor(max_workers=A.workers) as ex:
            for n, (k, v) in enumerate(ex.map(measure, jobs), 1):
                out[k] = v
                if n % 500 == 0:
                    print(f"    {n}/{len(jobs)}", flush=True)
        json.dump(out, open(A.out, "w"), indent=0)

        vals = [v for v in out.values() if v]
        dark = [v for v in vals if featureless(v)]
        print(f"\n  写出 {A.out}")
        if not vals:
            raise SystemExit("No measurable shot frames: all frame extractions failed or returned invalid data. "
                             f"Saved {len(out)} null measurements to {A.out}; no luminance evidence is available.")
        print(f"  可测 {len(vals)} 个，无内容 {len(dark)} 个 ({100*len(dark)/len(vals):.1f}%)")
        print(f"  测不了的 {sum(1 for v in out.values() if not v)} 个")


    # Run the scan only through the explicit command-line entry point.
    
    main()


if __name__ == "__main__":
    main()
