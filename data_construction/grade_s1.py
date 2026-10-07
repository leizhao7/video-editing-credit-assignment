"""Public construction draft: grade_s1.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/grade_s1.py [options]\n\n  --task  任务目录，默认取本脚本的上上级\n  --quiet\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    import argparse, json, os, sys

    P = argparse.ArgumentParser()
    P.add_argument("--task", default=None, help="任务目录，默认取本脚本的上上级")
    P.add_argument("--quiet", action="store_true")
    A = P.parse_args()

    T = A.task or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    key = json.load(open(f"{T}/verifier/answer.json"))
    gold = set(key["answer"])
    valid = {c["id"] for c in json.load(open(f"{T}/initial_state/manifest.json"))["candidates"]}

    p = f"{T}/output/selection.json"
    if not os.path.exists(p):
        print(json.dumps({"task": key["task"], "score": 0.0, "f1": 0.0, "error": "output/selection.json 不存在"},
                         ensure_ascii=False))
        sys.exit(0)
    try:
        raw = json.load(open(p)).get("selected")
    except Exception as e:
        print(json.dumps({"task": key["task"], "score": 0.0, "f1": 0.0, "error": f"JSON 解析失败: {e}"},
                         ensure_ascii=False))
        sys.exit(0)
    if not isinstance(raw, list):
        print(json.dumps({"task": key["task"], "score": 0.0, "f1": 0.0, "error": "selected 不是数组"},
                         ensure_ascii=False))
        sys.exit(0)

    seen, dup, unknown, sel = set(), [], [], set()
    for x in raw:
        x = str(x).strip()
        if x in seen:
            dup.append(x); continue
        seen.add(x)
        (sel if x in valid else unknown).add(x) if x in valid else unknown.append(x)

    hit = len(sel & gold)
    prec = hit / len(sel) if sel else 0.0
    rec = hit / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    out = {"task": key["task"], "score": round(f1, 4), "f1": round(f1, 4),
           "pass": f1 >= 0.8,
           "precision": round(prec, 4), "recall": round(rec, 4),
           "selected": len(sel), "answer": len(gold), "hit": hit,
           "missed": sorted(gold - sel), "wrong": sorted(sel - gold)}
    if unknown:
        out["unknown_ids"] = sorted(unknown)
    if dup:
        out["duplicates"] = sorted(set(dup))
    print(json.dumps(out, ensure_ascii=False, indent=1 if not A.quiet else None))


if __name__ == "__main__":
    main()
