"""Public construction draft: cut_takes.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/cut_takes.py [options]\n\n  --film\n  --all\n  --handle  台词前后各留几秒\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, glob, json, os, subprocess, warnings

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
        if not _ASR:
            from faster_whisper import WhisperModel
            _ASR.append(WhisperModel("small.en", device="cpu", compute_type="int8"))
        try:
            segs, _ = _ASR[0].transcribe(path)
        except Exception:
            return []
        out = []
        for s in segs:
            t = s.text.strip()
            if t:
                out.append({"s": round(s.start, 2), "e": round(s.end, 2), "text": t[:80]})
        return out


    def film_src(slug):
        pb = json.load(open(f"{ROOT}/exports/{slug}/probe.json"))
        c = [p for p in sorted(glob.glob(f'{pb["dir"]}/*'))
             if p.lower().endswith((".mp4", ".mkv", ".mov")) and "railer" not in p]
        return c[0] if c else None


    def cut_takes(slug):
        lo = f"{ROOT}/exports/{slug}/line_origins.json"
        src = film_src(slug)
        if not (os.path.exists(lo) and src):
            return None
        d = json.load(open(lo))
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
            if not os.path.exists(p):
                r = subprocess.run(
                    ["ffmpeg", "-v", "error", "-ss", f"{a:.3f}",
                     "-t", f"{(e - s) + head + A.handle:.3f}", "-i", src,
                     "-vn", "-ac", "2", "-ar", "48000", "-c:a", "aac", "-b:a", "128k",
                     "-y", p], capture_output=True, text=True)
                if r.returncode or not os.path.exists(p):
                    continue
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
        json.dump({"film": slug, "handle_s": A.handle, "n_takes": len(rows),
                   "n_lines": d["n_lines"], "n_found": d["n_found"], "takes": rows},
                  open(f"{ROOT}/exports/{slug}/takes.json", "w"),
                  ensure_ascii=False, indent=1)
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
