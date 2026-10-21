"""Public construction draft: grade_s2.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/grade_s2.py [options]\n\n  --task\n  --quiet\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    import argparse, bisect, json, os, sys

    P = argparse.ArgumentParser()
    P.add_argument("--task", default=None)
    P.add_argument("--quiet", action="store_true")
    A = P.parse_args()
    T = A.task or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    key = json.load(open(f"{T}/verifier/answer.json"))
    gold = key["order"]
    rank = {c: i for i, c in enumerate(gold)}

    def fail(msg):
        print(json.dumps({"task": key["task"], "score": 0.0, "pass": False, "error": msg},
                         ensure_ascii=False))
        sys.exit(0)

    p = f"{T}/output/order.json"
    if not os.path.exists(p):
        fail("output/order.json is missing")
    try:
        got = json.load(open(p)).get("order")
    except Exception as e:
        fail(f"JSON parsing failed: {e}")
    if not isinstance(got, list):
        fail("order must be an array")
    got = [str(x).strip() for x in got]
    if len(set(got)) != len(got):
        fail("Duplicate clip IDs")
    if set(got) != set(gold):
        miss = sorted(set(gold) - set(got)); extra = sorted(set(got) - set(gold))
        fail(f"Clip IDs do not match: missing {miss[:4]}, extra {extra[:4]}")

    n = len(got)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rank[got[i]], rank[got[j]]
            conc += a < b
            disc += a > b
    tot = conc + disc
    tau = (conc - disc) / tot if tot else 1.0

    # --- (1 - ND) * LIS * ADJ ---------------------------------------------------
    perm = [rank[c] for c in got]
    denom = (n * n) // 2
    nd = sum(abs(p - i) for i, p in enumerate(perm)) / denom if denom else 0.0

    def _lis(seq):
        tails = []
        for x in seq:
            i = bisect.bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return len(tails)

    lis = _lis(perm) / n if n else 0.0
    adj = (sum(1 for i in range(n - 1) if perm[i + 1] == perm[i] + 1) / (n - 1)
           if n > 1 else 1.0)
    score = (1 - nd) * lis * adj
    exact = got == gold
    # Retain tau as an additional diagnostic.
    print(json.dumps({"task": key["task"], "score": round(score, 4), "pass": score >= 0.5,
                      "nd": round(nd, 4), "lis": round(lis, 4), "adj": round(adj, 4),
                      "kendall_tau": round(tau, 4), "tau_score": round((tau + 1) / 2, 4),
                      "exact": exact,
                      "pairs_right": conc, "pairs_wrong": disc, "n_clips": n,
                      "your_order": got, "gold_order": gold},
                     ensure_ascii=False, indent=1 if not A.quiet else None))


if __name__ == "__main__":
    main()
