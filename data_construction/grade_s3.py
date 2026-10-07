"""Public construction draft: grade_s3.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/grade_s3.py [options]\n\n  --task\n  --quiet\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    import argparse, json, math, os, sys

    P = argparse.ArgumentParser()
    P.add_argument("--task", default=None)
    P.add_argument("--quiet", action="store_true")
    A = P.parse_args()
    T = A.task or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    key = json.load(open(f"{T}/verifier/answer.json"))
    # Compare seconds to seconds. The answer key converts each tolerance at its measured frame rate;
    # source, reference, and normalized-clip frame rates must not be interchanged here.


    FPS = key["fps"]; DTOL_S = key["dur_tol_s"]
    gold = {c["clip"]: c for c in key["cuts"]}
    # the clips' real lengths, so an out point past the end of the material is caught rather
    # than counted as a delivered duration
    lim = {c["id"]: c["clip_dur_s"]
           for c in json.load(open(f"{T}/initial_state/manifest.json"))["clips"]}

    def fail(msg):
        print(json.dumps({"task": key["task"], "score": 0.0, "pass": False, "error": msg},
                         ensure_ascii=False))
        sys.exit(0)

    p = f"{T}/output/cuts.json"
    if not os.path.exists(p):
        fail("output/cuts.json 不存在")
    try:
        got = json.load(open(p)).get("cuts")
    except Exception as e:
        fail(f"JSON 解析失败: {e}")
    if not isinstance(got, list):
        fail("cuts 不是数组")

    mine = {}
    for c in got:
        try:
            cid = str(c["clip"]).strip()
            if not cid:
                fail("clip id 为空")
            if cid in mine:
                fail(f"clip 重复: {cid}")
            a, b = float(c["in_s"]), float(c["out_s"])
            if not (math.isfinite(a) and math.isfinite(b)):
                fail(f"clip {cid} 的 in_s/out_s 不是有限数")
            mine[cid] = (a, b)
        except Exception:
            fail(f"条目缺 clip/in_s/out_s: {str(c)[:60]}")
    miss = [c for c in gold if c not in mine]
    extra = [c for c in mine if c not in gold]
    if miss or extra:
        fail(f"编号对不上：缺 {miss[:4]} 多 {extra[:4]}")

    rows, hit, scored = [], 0, 0
    tot_dur = 0.0
    for cid, g in gold.items():
        a, b = mine[cid]
        if b <= a or a < 0 or b > lim.get(cid, b) + 0.02:
            # A degenerate or out-of-range cut is a miss, not an exemption. Skipping it with
            # `continue` took the shot out of the denominator, so a submission could delete
            # every shot it was unsure of — out_s <= in_s — and score 100% on the remainder.
            # The out-of-range half matters for the same reason: an out point past the end of
            # the clip is a duration the agent cannot actually deliver.
            why = ("出点不在入点之后" if b <= a
                   else "入点为负" if a < 0 else f"出点超过素材长度 {lim.get(cid)}s")
            rows.append({"clip": cid, "scored": bool(g["scored"]), "hit": False, "error": why})
            scored += bool(g["scored"])
            continue
        tot_dur += b - a
        if not g["scored"]:
            rows.append({"clip": cid, "scored": False}); continue
        scored += 1
        din = abs(a - g["in_s"])
        ddur = abs((b - a) - g["dur_s"])
        ok = din <= g["in_tol_s"] and ddur <= DTOL_S
        hit += ok
        rows.append({"clip": cid, "scored": True, "hit": ok,
                     "in_err_s": round(din, 4), "in_tol_s": g["in_tol_s"],
                     "dur_err_s": round(ddur, 4), "dur_tol_s": DTOL_S,
                     "in_err_frames": round(din * FPS, 1),      # 仅供阅读
                     "plateau_frames": g["plateau_frames"]})

    # Every shot unscored means the chunk is all alternate takes — there was nothing to get
    # right. Reporting 0.0 reads as "failed" when the honest answer is "not gradable this way".
    score = hit / scored if scored else None
    tgt = key["target_duration_s"]
    print(json.dumps({"task": key["task"],
                      "score": None if score is None else round(score, 4),
                      "pass": None if score is None else score >= 0.7,
                      "scorable": score is not None,
                      "hit": hit, "scored": scored, "n_clips": len(gold),
                      "total_duration_s": round(tot_dur, 3), "target_duration_s": tgt,
                      "duration_err_s": round(tot_dur - tgt, 3),
                      "cards_excluded_s": key["cards_excluded_s"],
                      "per_clip": rows},
                     ensure_ascii=False, indent=1 if not A.quiet else None))


if __name__ == "__main__":
    main()
