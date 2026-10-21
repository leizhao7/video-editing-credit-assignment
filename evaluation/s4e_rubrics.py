"""Continuous-video rubrics; legacy aggregation weights and soft penalty preserved."""
from rubrics_video import S4 as RUBRICS
SOFT = 0.75
NEVER_SHOW = ("golden", "reference", "answer")


def aggregate(scores):
    """Aggregate four 0–4 rubric bands into a 0–1 score.

    Apply a multiplicative 0.75 penalty for each zero-scored rubric.
    """
    got = {r["id"]: scores.get(r["id"]) for r in RUBRICS}
    have = {k: v for k, v in got.items() if isinstance(v, (int, float))}
    if not have:
        return {"score": None, "why": "No rubric received a score"}
    wsum = sum(r["weight"] for r in RUBRICS if r["id"] in have)
    raw = sum(have[r["id"]] * r["weight"] for r in RUBRICS if r["id"] in have) / wsum / 4.0
    zeros = [k for k, v in have.items() if v == 0]
    score = raw * (0.75 ** len(zeros))
    return {"score": round(score, 3), "raw": round(raw, 3),
            "per_rubric": have,
            "missing": [k for k, v in got.items() if v is None],
            "penalty": f"{len(zeros)} zero-scored rubrics ({','.join(zeros)}); multiply by {0.75 ** len(zeros):.2f}"
                       if zeros else None}
