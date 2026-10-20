"""Public construction draft: cut_takes.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/cut_takes.py [options]\n\n  --film\n  --all\n  --handle  台词前后各留几秒\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)


def film_src(slug, root):
    """Resolve only the raw source recorded in this film's probe metadata."""
    import json
    from pathlib import Path
    try:
        from .release_config import validate_data_slug, export_slug
    except ImportError:
        from release_config import validate_data_slug, export_slug

    slug = validate_data_slug(root, slug)
    probe_path = Path(root) / "exports" / slug / "probe.json"
    if not probe_path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Probe path escapes the private data root")
    with probe_path.open(encoding="utf-8") as stream:
        probe = json.load(stream)
    if not isinstance(probe, dict):
        raise ValueError("Probe metadata must be an object")
    export_slug(probe, probe_path)
    source = probe.get("source")
    if (not isinstance(source, str) or not source.strip() or source in {".", ".."}
            or any(char in source for char in ("/", "\\", "\0"))):
        raise ValueError("Probe source must be a single safe filename")
    directory = probe.get("dir")
    if not isinstance(directory, str) or not directory.strip() or "\0" in directory:
        raise ValueError("Probe dir must name the source input directory")
    path = Path(directory).expanduser() / source
    # User-owned source media may be a symlink to storage outside the data root.
    if not path.is_file():
        raise FileNotFoundError(f"Source named by probe.json does not exist: {path}")
    return str(path)


def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook, validate_data_slug)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook, validate_data_slug)

    import argparse, glob, json, os, subprocess, warnings
    from pathlib import Path
    from tempfile import TemporaryDirectory

    ROOT = DATA_ROOT


    P = argparse.ArgumentParser()
    P.add_argument("--film")
    P.add_argument("--all", action="store_true")
    P.add_argument("--handle", type=float, default=3.0, help="台词前后各留几秒")
    A = P.parse_args()
    require_target(P, A)


    _ASR = []


    def speech_spans(path):
        """文件里实际有人说话的区间。定位窗口只覆盖匹配上的那一句，不够用。"""
        warnings.filterwarnings("ignore")
        try:
            if not _ASR:
                from faster_whisper import WhisperModel
                _ASR.append(WhisperModel("small.en", device="cpu", compute_type="int8"))
            segs, _ = _ASR[0].transcribe(path)
            out = []
            for s in segs:
                t = s.text.strip()
                if t:
                    out.append({"s": round(s.start, 2), "e": round(s.end, 2), "text": t[:80]})
            return out
        except Exception as exc:
            raise RuntimeError(
                f"Speech transcription failed for {path}: {exc}. "
                "Install compatible dependencies with "
                "`python -m pip install -r data_construction/requirements.txt` and retry."
            ) from exc


    def cut_takes(slug):
        slug = validate_data_slug(ROOT, slug)
        result_path = Path(ROOT) / "exports" / slug / "takes.json"
        # A failed attempt must not leave an earlier manifest looking successful.
        result_path.unlink(missing_ok=True)
        lo = f"{ROOT}/exports/{slug}/line_origins.json"
        src = film_src(slug, ROOT)
        if not (os.path.exists(lo) and src):
            return None
        with open(lo, encoding="utf-8") as stream:
            d = json.load(stream)
        out = f"{ROOT}/exports/{slug}/takes"
        os.makedirs(out, exist_ok=True)
        rows, n_cut = [], 0
        for i, ln in enumerate(d["lines"], 1):
            if not ln.get("found"):
                continue
            s, e = ln["film_s"], ln["film_e"]
            a = max(0.0, s - A.handle)
            head = s - a
            pid = f"t{i:02d}"
            p = f"{out}/{pid}.m4a"
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                with TemporaryDirectory(prefix=f".{pid}-", dir=out) as temporary:
                    candidate = os.path.join(temporary, "take.m4a")
                    r = subprocess.run(
                        ["ffmpeg", "-v", "error", "-ss", f"{a:.3f}",
                         "-t", f"{(e - s) + head + A.handle:.3f}", "-i", src,
                         "-vn", "-ac", "2", "-ar", "48000", "-c:a", "aac", "-b:a", "128k",
                         "-y", candidate], capture_output=True, text=True)
                    if r.returncode:
                        raise RuntimeError(f"Audio conversion failed for {slug}/{pid} "
                                           f"(ffmpeg exit {r.returncode}): {r.stderr.strip()}")
                    if not os.path.isfile(candidate) or os.path.getsize(candidate) == 0:
                        raise RuntimeError(f"Audio conversion produced no output for {slug}/{pid}")
                    os.replace(candidate, p)
            # Transcribe the full extracted take, including handles, so its usable speech intervals match the file.


            spans = speech_spans(p)
            # Exclude takes with no intelligible transcript rather than treating audio energy as evidence of speech.
            
            
            if not spans:
                continue
            n_cut += 1
            rows.append({"id": pid, "file": f"takes/{pid}.m4a",
                         "speech_spans_s": spans,
                         # 预告片里那句话说了什么，和它在成片里的位置 —— 出题时不给 agent，
                         # 这是建题用的账
                         "trailer_text": ln["text"],
                         "trailer_s": ln["s"], "trailer_e": ln["e"],
                         "film_s": round(s, 3), "film_e": round(e, 3),
                         "film_text": ln.get("film_text", ""),
                         "match_score": ln.get("score"),
                         "words_start_at_s": round(head, 3),
                         "words_dur_s": round(e - s, 3),
                         "take_dur_s": round((e - s) + head + A.handle, 3)})
        with TemporaryDirectory(prefix=".takes-metadata-", dir=result_path.parent) as temporary:
            candidate = Path(temporary) / "takes.json"
            with candidate.open("w", encoding="utf-8") as stream:
                json.dump({"film": slug, "handle_s": A.handle, "n_takes": len(rows),
                           "n_lines": d["n_lines"], "n_found": d["n_found"], "takes": rows},
                          stream, ensure_ascii=False, indent=1)
            os.replace(candidate, result_path)
        return len(rows), d["n_lines"]


    films = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(f"{ROOT}/exports/*/line_origins.json")) \
        if A.all else [A.film]
    T = O = 0
    for slug in films:
        r = cut_takes(slug)
        if not r:
            print(f"  {slug:<24} 缺 line_origins 或正片，跳过")
            continue
        n, tot = r
        T += tot
        O += n
        print(f"  {slug:<24} 切出 {n:3d} 条原始录音 / {tot:3d} 句", flush=True)
    print(f"\n合计 {O}/{T}")


if __name__ == "__main__":
    main()
