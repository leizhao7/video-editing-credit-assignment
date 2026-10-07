"""Public construction draft: locate_lines.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/locate_lines.py [options]\n\n  --film\n  --all\n  --gap  合并成一段语流的最大词间隔\n  --floor  判定找到的最低分\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, glob, json, os, re, sys

    ROOT = DATA_ROOT


    P = argparse.ArgumentParser()
    P.add_argument("--film")
    P.add_argument("--all", action="store_true")
    P.add_argument("--gap", type=float, default=0.35, help="合并成一段语流的最大词间隔")
    P.add_argument("--floor", type=float, default=0.55, help="判定找到的最低分")
    A = P.parse_args()
    require_target(P, A)

    STOP = {"a", "the", "is", "of", "to", "and", "i", "it", "you", "that", "in", "we",
            "s", "t", "on", "at", "for", "this", "be", "was", "are", "he", "she"}
    tok = lambda s: [w for w in re.findall(r"[a-z']+", s.lower()) if w]


    def runs_from(path, gap):
        """词表 → 语流。间隔 ≤gap 的词属于同一段话，这和答案里 J/L 的口径一致。"""
        tr = json.load(open(path))
        W = sorted([w for s in tr["segments"] for w in s["words"] if w.get("p", 1) > 0.4],
                   key=lambda w: w["s"])
        out, cur = [], None
        for w in W:
            if cur and w["s"] - cur[1] <= gap:
                cur = (cur[0], max(cur[1], w["e"]), cur[2] + [w["w"]])
            else:
                if cur:
                    out.append(cur)
                cur = (w["s"], w["e"], [w["w"]])
        if cur:
            out.append(cur)
        return [{"s": a, "e": b, "text": " ".join(t)} for a, b, t in out]


    def score(q, w):
        """有序重合：相同的词算分，顺序也对的加权。停用词不计，否则短句全靠 the/a 撞上。"""
        qs = [x for x in q if x not in STOP]
        ws = [x for x in w if x not in STOP]
        if not qs:
            return 0.0
        have = list(ws)
        hit = 0
        for x in qs:
            if x in have:
                have.remove(x)
                hit += 1
        cov = hit / len(qs)
        # 顺序奖励：最长公共子序列 / 查询长度
        m, n = len(qs), len(ws)
        if m and n:
            prev = [0] * (n + 1)
            for i in range(1, m + 1):
                cur = [0] * (n + 1)
                for j in range(1, n + 1):
                    cur[j] = prev[j - 1] + 1 if qs[i - 1] == ws[j - 1] else max(prev[j], cur[j - 1])
                prev = cur
            order = prev[n] / m
        else:
            order = 0.0
        return 0.6 * cov + 0.4 * order


    def locate(slug):
        gp = f"{ROOT}/exports/{slug}/golden_transcript.json"
        sp = f"{ROOT}/exports/{slug}/source_transcript.json"
        if not (os.path.exists(gp) and os.path.exists(sp)):
            return None
        lines = runs_from(gp, A.gap)
        src = json.load(open(sp))
        SW = sorted([w for s in src["segments"] for w in s["words"] if w.get("p", 1) > 0.4],
                    key=lambda w: w["s"])
        words = [w["w"] for w in SW]
        low = [w.lower().strip("',.!?") for w in words]

        rows = []
        for ln in lines:
            q = tok(ln["text"])
            if len(q) < 2:
                rows.append({**ln, "found": False, "why": "太短，不判"})
                continue
            win = max(3, len(q))
            # Cap elapsed source time for a word window so long silent gaps cannot create implausible matches.
            
            
            span_cap = (ln["e"] - ln["s"]) * 3.0 + 3.0
            best, bi = 0.0, -1
            # 窗口在 0.7×~1.6× 查询长度之间滑动，一句台词在正片里可能被停顿切开
            for L in {max(3, int(win * 0.7)), win, int(win * 1.2) + 1, int(win * 1.6) + 1}:
                for i in range(0, max(1, len(low) - L + 1)):
                    j = min(i + L - 1, len(SW) - 1)
                    if SW[j]["e"] - SW[i]["s"] > span_cap:
                        continue
                    sc = score(q, low[i:i + L])
                    if sc > best:
                        best, bi, bl = sc, i, L
            if best < A.floor or bi < 0:
                # When the whole line does not match, look for a contiguous shared word sequence
                # that can serve as a partial source anchor.
                
                
                anc = None
                for L in range(min(8, len(q)), 2, -1):
                    for i in range(len(q) - L + 1):
                        g = q[i:i + L]
                        for j in range(len(low) - L + 1):
                            if low[j:j + L] != g:
                                continue
                            k = min(j + L - 1, len(SW) - 1)
                            if SW[k]["e"] - SW[j]["s"] > span_cap:
                                continue
                            anc = (j, L, L / len(q))
                            break
                        if anc:
                            break
                    if anc:
                        break
                if anc:
                    j, L, cov = anc
                    rows.append({**ln, "found": True, "score": round(cov, 3),
                                 "via": "n-gram 锚点",
                                 "anchor_words": L,
                                 "film_s": round(SW[j]["s"], 3),
                                 "film_e": round(SW[min(j + L - 1, len(SW) - 1)]["e"], 3),
                                 "film_text": " ".join(words[j:j + L])})
                    continue
                rows.append({**ln, "found": False, "best": round(best, 3),
                             "why": "正片里找不到这句"})
                continue
            rows.append({**ln, "found": True, "score": round(best, 3),
                         "film_s": round(SW[bi]["s"], 3),
                         "film_e": round(SW[min(bi + bl - 1, len(SW) - 1)]["e"], 3),
                         "film_text": " ".join(words[bi:bi + bl])})
        n_ok = sum(1 for r in rows if r.get("found"))
        json.dump({"film": slug, "n_lines": len(rows), "n_found": n_ok,
                   "floor": A.floor, "lines": rows},
                  open(f"{ROOT}/exports/{slug}/line_origins.json", "w"),
                  ensure_ascii=False, indent=1)
        return len(rows), n_ok


    films = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(f"{ROOT}/exports/*/source_transcript.json")) \
        if A.all else [A.film]
    T = O = 0
    for slug in films:
        r = locate(slug)
        if not r:
            print(f"  {slug:<24} 缺转录，跳过")
            continue
        n, ok = r
        T += n
        O += ok
        print(f"  {slug:<24} {ok:3d}/{n:3d} 句在正片里找到  {ok/max(n,1):.0%}", flush=True)
    print(f"\n合计 {O}/{T}  {O/max(T,1):.0%}")


if __name__ == "__main__":
    main()
