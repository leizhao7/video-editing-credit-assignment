"""Public construction draft: build_report.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/build_report.py [options]\n\n  --film\n  --win\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, csv, json, os, re, subprocess
    from collections import Counter


    # Keep artifact-generating work inside the explicit entry point; importing must not run it.


    ROOT = DATA_ROOT
    P = argparse.ArgumentParser()
    P.add_argument("--film", required=True)
    P.add_argument("--win", type=float, default=1.8)
    A = P.parse_args()
    E, T = f"{ROOT}/exports/{A.film}", f"{ROOT}/edit/task_{A.film}"
    pb = json.load(open(f"{E}/probe.json"))
    G = f"{pb['dir']}/{pb['golden']}"
    CROP = (pb.get("crop_expr", "") + ",") if pb.get("crop_expr") else (
        f"crop=iw:iw/{pb['crop']}:0:(ih-iw/{pb['crop']})/2," if pb["crop"] else "")


    def load(p, d=None):
        return json.load(open(p)) if os.path.exists(p) else d


    # Use video-stream duration; container duration can extend beyond the last decodable frame.
    
    
    GDUR = pb["gold"].get("vdur") or pb["gold"]["dur"]
    GFPS = pb["gold"]["fps"]
    MISSING = []
    # Regenerate stills when the cut list is newer; a stable cut index does not imply stable timing.
    
    
    _CLT = os.path.getmtime(f"{T}/stages/cut_list.json") if os.path.exists(
        f"{T}/stages/cut_list.json") else 0


    def _fresh(o):
        return os.path.exists(o) and os.path.getmtime(o) >= _CLT
    cl = load(f"{T}/stages/cut_list.json", {}).get("cuts")
    if not cl:
        gm = list(csv.DictReader(open(f"{E}/golden_clips/manifest.csv")))
        cl = [{"idx": i + 1, "after": a["id"], "before": b["id"],
               "timeline_s": float(b["start_s"])} for i, (a, b) in enumerate(zip(gm, gm[1:]))]
    au = {c["idx"]: c for c in (load(f"{E}/audio_cuts.json", {}).get("cuts") or [])}
    # Include measured picture continuity alongside the audio evidence used to review each cut.
    
    
    pm = {c["idx"]: c for c in (load(f"{T}/stages/picture_measured.json", {}).get("cuts") or [])}
    s4 = {c["idx"]: c for c in (load(f"{T}/stages/s4_cuts.json", {}).get("cuts") or [])}


    def _vocab(md):
        """Read vocabulary tables from the prompt used for this annotation run.

        Keep legacy Chinese heading aliases for previously generated private
        reports. New prompts and report labels use English.
        """
        out = {}
        aliases = {
            "transition": ("### Transition vocabulary", "### 一、"),
            "technique": ("### Technique vocabulary", "### 二、"),
            "shared": ("## Shared vocabulary", "## 前面几部片已经造过的词"),
        }
        for key, headings in aliases.items():
            positions = [md.find(head) for head in headings if md.find(head) >= 0]
            if not positions:
                continue
            start = md.find("\n", min(positions))
            if start < 0:
                continue
            # Keep shared extensions distinct from built-in labels, and stop
            # each vocabulary table at the next section or subsection heading.
            following = re.search(r"^#{2,3} ", md[start + 1:], re.M)
            end = start + 1 + following.start() if following else len(md)
            out[key] = set(re.findall(r"^\|\s*`([a-z_]+)`", md[start:end], re.M))
        return out


    # Use the prompt saved for this annotation run, whose vocabulary may differ from the current template.


    import glob as _g
    _runs = sorted(_g.glob(f"{ROOT}/edit/runs/{A.film}/*/PROMPT.md"), key=os.path.getmtime)
    PROMPT = _runs[-1] if _runs else f"{CODE_ROOT}/prompts/PROMPT_S4.md"
    KNOWN = _vocab(open(PROMPT).read()) if os.path.exists(PROMPT) else {}
    novel = {}
    SHARED = KNOWN.get("shared", set())
    reused = {}
    for dim, field in [("transition", lambda c: (c.get("transition") or {}).get("type")),
                       ("technique", lambda c: c.get("technique"))]:
        seen = {}
        for c in s4.values():
            v = field(c)
            if v and v in SHARED and v not in KNOWN.get(dim, ()):
                reused[v] = reused.get(v, 0) + 1
            if not v or v in KNOWN.get(dim, ()) or v in SHARED: continue
            e = seen.setdefault(v, {"n": 0, "idx": [], "eg": ""})
            e["n"] += 1; e["idx"].append(c.get("idx"))
            if not e["eg"]: e["eg"] = (c.get("note") or "")[:120]
        novel[dim] = dict(sorted(seen.items(), key=lambda kv: -kv[1]["n"]))
    if s4:
        os.makedirs(f"{T}/stages", exist_ok=True)
        json.dump({"stage": "S4", "film": A.film,
                   "known": {k: sorted(v) for k, v in KNOWN.items() if k != "shared"},
                   "reused_from_corpus": dict(sorted(reused.items(), key=lambda kv: -kv[1])),
                   **novel}, open(f"{T}/stages/s4_novel_terms.json", "w"),
                  ensure_ascii=False, indent=1)
    NOVEL = set(novel.get("technique", {}))
    NOVEL_TR = set(novel.get("transition", {}))

    os.makedirs(f"{E}/cutframes", exist_ok=True); os.makedirs(f"{E}/cutclips", exist_ok=True)
    for c in cl:
        i, t = c["idx"], c["timeline_s"]
        # Clamp sampling to the last decodable frame so late cuts still produce valid images.
        
        
        last = GDUR - 2.0 / max(GFPS, 1.0)
        for tag, tt in [("a", max(0.0, t - 0.10)), ("b", min(t + 0.10, last))]:
            o = f"{E}/cutframes/c{i:03d}{tag}.jpg"
            if not _fresh(o):
                # Delete stale images before grabbing a new one so existence checks cannot pass on old output.
                
                
                if os.path.exists(o): os.remove(o)
                subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, tt):.3f}", "-i", G,
                                "-frames:v", "1", "-vf", CROP + "scale=320:-2",
                                "-q:v", "5", "-y", o], check=False)
                if not os.path.exists(o):        # Retry earlier; report any frame still missing.
                    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, tt - 0.20):.3f}",
                                    "-i", G, "-frames:v", "1", "-vf", CROP + "scale=320:-2",
                                    "-q:v", "5", "-y", o], check=False)
                if not os.path.exists(o):
                    MISSING.append(f"c{i:03d}{tag}@{tt:.2f}s")
        o = f"{E}/cutclips/c{i:03d}.mp4"
        if not _fresh(o):
            if os.path.exists(o): os.remove(o)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0,t-A.win):.3f}",
                            "-t", f"{A.win*2:.3f}", "-i", G,
                            "-vf", CROP + "scale=480:-2", "-c:v", "libx264", "-crf", "28",
                            "-preset", "veryfast", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", o],
                           check=False)

    rows = []
    for c in cl:
        i = c["idx"]; a = au.get(i, {}); s = s4.get(i, {})
        tr = s.get("transition") or {}; sp = a.get("speech") or {}
        rows.append({"i": i, "t": round(c["timeline_s"], 2), "A": c["after"], "B": c["before"],
                     "tr": tr.get("type", "cut"),
                     "trp": {k: v for k, v in tr.items() if k != "type" and v},
                     "tc": s.get("technique", ""), "note": s.get("note", ""),
                     "sp": sp.get("type", "hard"), "off": sp.get("offset"),
                     "line": sp.get("line", ""), "mu": a.get("music", []),
                     "pc": (pm.get(i) or {}).get("continuous"),
                     "pcf": bool((pm.get(i) or {}).get("flat"))})

    sp = Counter(r["sp"] for r in rows); mu = Counter(x for r in rows for x in r["mu"])
    tc = Counter(r["tc"] for r in rows if r["tc"]); tr = Counter(r["tr"] for r in rows)
    CSS = open(f"{CODE_ROOT}/assets/report.css").read() if os.path.exists(f"{CODE_ROOT}/assets/report.css") else ""
    JS = open(f"{CODE_ROOT}/assets/report.js").read()
    JS = JS.replace("__ROWS__", json.dumps(rows, ensure_ascii=False)) \
           .replace("__NOVEL__", json.dumps(sorted(NOVEL))) \
           .replace("__NOVELTR__", json.dumps(sorted(NOVEL_TR)))
    btn = [('all', f'All {len(rows)}'), ('J', f'J-cut ({sp["J"]})'), ('L', f'L-cut ({sp["L"]})'),
           ('hit', f'Music accents ({mu["hit"]})'), ('change', f'Music changes ({mu["change"]})'),
           ('bridge', f'Music bridges ({mu["bridge"]})'), ('dip', f'Other transitions ({len(rows)-tr["cut"]})'),
           ('pc', f'Continuous picture ({sum(1 for r in rows if r["pc"])})')]
    if NOVEL: btn.append(('nv', f'New techniques ({sum(tc[k] for k in NOVEL)})'))
    if NOVEL_TR: btn.append(('nvt', f'New transitions ({sum(tr[k] for k in NOVEL_TR)})'))
    html = (f'<meta charset="utf-8"><title>{A.film} · {len(rows)} cut boundaries</title>\n<style>\n{CSS}\n</style>\n'
            f'<header><h1>{A.film} · Cut analysis: {len(rows)} boundaries</h1>\n<div class="s">'
            f'Speech hard <b>{sp["hard"]}</b> · J <b>{sp["J"]}</b> · L <b>{sp["L"]}</b> &nbsp;|&nbsp; '
            f'Music hit <b>{mu["hit"]}</b> · change <b>{mu["change"]}</b> · '
            f'swell <b>{mu["swell"]}</b> · bridge <b>{mu["bridge"]}</b>'
            + (f' &nbsp;|&nbsp; {len(tc)} techniques' if tc else ' &nbsp;|&nbsp; <i>No S4 annotations; audio analysis only</i>')
            + '</div>\n<div class="f">'
            + "".join(f'<button{" class=on" if k=="all" else ""} data-f="{k}">{l}</button>' for k, l in btn)
            + '</div></header>\n<div class="grid" id="g"></div>\n<script>\n' + JS + '\n</script>')
    open(f"{E}/cuts_report.html", "w").write(html)
    print(f"{A.film}: {len(rows)} cut boundaries; speech J={sp['J']}/L={sp['L']}/hard={sp['hard']}; "
          f"music hit={mu['hit']}/change={mu['change']}/bridge={mu['bridge']}"
          + (f"; {len(tc)} techniques" if tc else "; no S4 annotations"))
    if MISSING:
        print(f"Warning: could not extract {len(MISSING)} frames: {MISSING[:6]}")
    print(f"-> {E}/cuts_report.html")


if __name__ == "__main__":
    main()
