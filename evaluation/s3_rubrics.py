"""Continuous-video rubrics; legacy aggregation weights and soft penalty preserved."""
from rubrics_video import S3 as RUBRICS
SOFT = 0.75
NEVER_SHOW = ("golden", "reference", "answer")


def aggregate(scores):
    """三个 0–4 合成一个 0–1。

    加权平均，任何一条拿 0 则乘 0.75，一条一次。

    Soft penalties preserve differences in the weighted aggregate.
    """
    got = {r["id"]: scores.get(r["id"]) for r in RUBRICS}
    have = {k: v for k, v in got.items() if isinstance(v, (int, float))}
    if not have:
        return {"score": None, "why": "没有任何一条评出分"}
    wsum = sum(r["weight"] for r in RUBRICS if r["id"] in have)
    raw = sum(have[r["id"]] * r["weight"] for r in RUBRICS if r["id"] in have) / wsum / 4.0
    zeros = [k for k, v in have.items() if v == 0]
    score = raw * (SOFT ** len(zeros))
    return {"score": round(score, 3), "raw": round(raw, 3),
            "per_rubric": have, "missing": [k for k, v in got.items() if v is None],
            "penalty": (f"{len(zeros)} 条拿 0（{','.join(zeros)}），乘 "
                        f"{SOFT ** len(zeros):.2f}") if zeros else None}
