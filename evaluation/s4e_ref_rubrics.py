"""Continuous-video rubrics; legacy aggregation weights and soft penalty preserved."""
from rubrics_video import S4_REFERENCE as RUBRICS
SOFT = 0.75
NEVER_SHOW = ("golden", "reference", "answer")


def aggregate(scores):
    """三个 0–4 合成一个 0–1。跟盲判同一套软惩罚：任一条拿 0，总分乘 0.75。"""
    got = {r["id"]: scores.get(r["id"]) for r in RUBRICS}
    have = {k: v for k, v in got.items() if isinstance(v, (int, float))}
    if not have:
        return {"score": None, "why": "没有任何一条评出分"}
    wsum = sum(r["weight"] for r in RUBRICS if r["id"] in have)
    raw = sum(have[r["id"]] * r["weight"] for r in RUBRICS if r["id"] in have) / wsum / 4.0
    zeros = [k for k, v in have.items() if v == 0]
    return {"score": round(raw * (0.75 ** len(zeros)), 3), "raw": round(raw, 3),
            "per_rubric": have,
            "missing": [k for k, v in got.items() if v is None],
            "penalty": f"{len(zeros)} 条拿 0（{','.join(zeros)}），乘 {0.75 ** len(zeros):.2f}"
                       if zeros else None}
