"""Public construction draft: card_text.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/card_text.py [options]\n\n  --film\n  --all\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, glob, json, os, re, subprocess, sys

    ROOT = DATA_ROOT


    P = argparse.ArgumentParser()
    P.add_argument("--film"); P.add_argument("--all", action="store_true")
    A = P.parse_args()
    require_target(P, A)


    TMP = os.environ.get("TMPDIR", "/tmp").rstrip("/")


    def ocr_text(png, invert=False):
        """Read card text using Tesseract plain-text output.
        Word-level confidence and layout boxes are not used to filter display typography."""
        src = png
        if invert:
            src = f"{TMP}/_cardinv_{os.path.basename(png)}"
            subprocess.run(["ffmpeg", "-v", "error", "-i", png, "-vf", "negate", "-y", src],
                           check=False)
            if not os.path.exists(src):
                return []
        r = subprocess.run(["tesseract", src, "stdout", "--psm", "12", "-l", "eng"],
                           capture_output=True)
        if invert:
            try: os.remove(src)
            except OSError: pass
        out = []
        for ln in r.stdout.decode("utf-8", "replace").splitlines():
            t = " ".join(ln.split())
            if t and re.search(r"[A-Za-z0-9]", t) and len(t) > 1:
                out.append(t)
        return out


    def read_card(png, W, H):
        """as-is and inverted; whichever gives more readable text wins"""
        a, b = ocr_text(png, False), ocr_text(png, True)

        def weight(ls):
            # letters, not lines — a pass returning three fragments of noise should lose to
            # one returning a single clean word
            return sum(len(re.sub(r"[^A-Za-z0-9]", "", x)) for x in ls)

        lines, inv = (a, False) if weight(a) >= weight(b) else (b, True)
        # Reject fragmented OCR text rather than using noise as the required card wording.


        toks = [t for ln in lines for t in re.findall(r"[A-Za-z0-9]+", ln)]
        real = [t for t in toks if len(t) >= 4]
        if not lines or not real or len(real) < max(1, len(toks) // 3):
            return {"text": None, "note": "Text could not be read",
                    "raw": " / ".join(lines) if lines else None}
        return {"lines": lines, "inverted": inv, "text": " / ".join(lines)}


    def dims(png):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", png],
                           capture_output=True, text=True).stdout.strip().split(",")
        return (int(r[0]), int(r[1])) if len(r) >= 2 else (1920, 1080)


    films = sorted(os.path.basename(os.path.dirname(p))[5:]
                   for p in glob.glob(f"{ROOT}/edit/task_*/s3_units.json")) if A.all else [A.film]
    for slug in films:
        T = f"{ROOT}/edit/task_{slug}"
        if not os.path.exists(f"{T}/s3_units.json"):
            continue
        units = json.load(open(f"{T}/s3_units.json"))["units"]
        out, nread = {}, 0
        for u in units:
            if u["kind"] != "card":
                continue
            states = []
            for st in u.get("stills", []):
                png = f"{T}/cards/{st}"
                if not os.path.exists(png):
                    continue
                r = read_card(png, *dims(png))
                r["still"] = st
                states.append(r)
                nread += bool(r.get("text"))
            out[u["id"]] = {"dur_s": round(u["dur"], 3), "n_states": u.get("n_states"),
                            "bg": u.get("bg"), "states": states}
        json.dump({"film": slug, "n_cards": len(out), "cards": out},
                  open(f"{ROOT}/exports/{slug}/card_text.json", "w"),
                  ensure_ascii=False, indent=1)
        tot = sum(len(v["states"]) for v in out.values())
        print(f"  {slug:<24} {len(out):3d} cards; {tot:3d} states; text read from {nread}", flush=True)


if __name__ == "__main__":
    main()
