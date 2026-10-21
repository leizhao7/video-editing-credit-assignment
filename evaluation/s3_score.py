"""Combine two S3 judgements and the deterministic golden-distance score.

    final = 0.2 × blind + 0.4 × referenced + 0.4 × deterministic

Two LLM readings of the same cut, plus a deterministic task-verifier distance.
The 0.2 blind weight preserves room for a defensible alternate edit; the two
reference-backed components receive 0.4 each to reduce judge drift.

The blind pass (`s3_judge.py`) asks whether the cut works on its own terms.
Joint consistency review corrects factual errors before aggregation; its blind
corrections must be grounded in task/submission evidence, not reference similarity.

The referenced pass (`s3_judge_ref.py`) uses the reference edit to assess what the
footage supports, including moment selection and the time a gesture needs.
It contributes 0.4 to the final score.

All three components must exist.  The deterministic component is the task's
own verifier: it compares submitted per-clip in-points and kept durations to
the golden edit using each shot's measured tolerance.  A missing component is
reported instead of silently renormalising the remaining weights.

    python3 s3_score.py --tasks /tmp/s3_batch.json
"""
import argparse, json, os, subprocess, sys

if __name__ != "__main__":
    raise ImportError(f"{__file__} is a command-line script that writes files; it cannot be imported")

from score_cli import add_task_arguments, validate_weights, task_paths, read_component, evaluate_tasks

P = argparse.ArgumentParser()
add_task_arguments(P)
P.add_argument("--w-blind", type=float, default=0.2)
P.add_argument("--w-ref", type=float, default=0.4)
P.add_argument("--w-deterministic", type=float, default=0.4)
P.add_argument("--quiet", action="store_true")
A = P.parse_args()
validate_weights(P, (A.w_blind, A.w_ref, A.w_deterministic))
tasks = task_paths(P, A)


def deterministic_score(task):
    """Run the immutable task verifier and retain its complete distance detail."""
    script = os.environ.get("S12_S3_DETERMINISTIC_GRADER", "")
    if not script:
        installed = os.path.join(os.path.dirname(__file__), "grade_s3.py")
        script = installed if os.path.isfile(installed) else f"{task}/verifier/grade.py"
    if not os.path.isfile(script):
        return None, {"error": "verifier/grade.py missing"}
    from score_cli import run_verifier
    score, detail, _ = run_verifier(script, task, quiet=True)
    return score, detail


def combine(task):
    from grading_contract import validate_task, InvalidSubmission, valid_score, judge_only, DIRECT_JUDGE_CONTRACT
    try:
        validate_task(task, "s_3")
    except InvalidSubmission as exc:
        out = {"task": task, "final": 0.0, "invalid_submission": True, "reason": str(exc)}
        return out
    name = task.split("training_task/")[-1]
    bp = f"{task}/evidence/s3_score.json"
    rp = f"{task}/evidence/s3_score_ref.json"
    b = read_component(bp)
    r = read_component(rp)
    d_score, d_detail = deterministic_score(task)
    miss = [n for n, x in (("blind", b), ("ref", r))
            if x is None or not valid_score(x.get("score")) or (x.get("partial") and not judge_only())]
    if not valid_score(d_score):
        miss.append("deterministic")
    if miss:
        return {"task": name, "final": None, "missing": miss,
                "deterministic": d_detail,
                "why": f"Missing {', '.join(miss)}; no aggregate is produced. "
                       "Weights are not renormalized when components are missing."}
    if judge_only():
        VERSION = DIRECT_JUDGE_CONTRACT
    else:
        from grading_consistency import approved_components, ReviewError, VERSION
        try:
            reviewed = approved_components(task, "s_3")
        except ReviewError as exc:
            out = {"task": name, "final": None, "missing": ["consistency"], "why": str(exc)}
            return out
        b, r = reviewed["blind"], reviewed["referenced"]
    weight_sum = A.w_blind + A.w_ref + A.w_deterministic
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"S3 weights must sum to 1, got {weight_sum}")
    fin = (A.w_blind * b["score"] + A.w_ref * r["score"]
           + A.w_deterministic * d_score)
    out = {"task": name, "final": round(fin, 3),
           "consistency_contract": VERSION,
           "judgement_mode": "direct-judge" if judge_only() else "jointly-reconciled",
           "initial_scores": {"blind": b.get("initial_score", b["score"]),
                              "referenced": r.get("initial_score", r["score"])},
           "blind": {"score": b["score"], "per_rubric": b.get("per_rubric"),
                     "weight": A.w_blind},
           "referenced": {"score": r["score"], "per_rubric": r.get("per_rubric"),
                          "weight": A.w_ref, "duration": r.get("duration", {}).get("band")},
           "deterministic": {"score": d_score, "weight": A.w_deterministic,
                             "metric": "fraction within golden in-point AND duration tolerance",
                             "detail": d_detail},
           "weight_sum": weight_sum,
           "spread": round(abs(b["score"] - r["score"]), 3),
           "note": ("Initial blind/reference judgments are used directly without an additional consistency review; "
                    "submission validation follows the active policy. " if judge_only() else
                    "Initial judgments are retained. Blind/reference components are jointly reviewed against video evidence; "
                    "blind scores are not adjusted for similarity to the reference. ") +
                   "The deterministic component is the task verifier's distance score comparing submitted cuts "
                   "with the golden in-points and durations."}
    return out


rows = evaluate_tasks(tasks, combine, lambda task: f"{task}/evidence/s3_final.json")
ok = [x for x in rows if x["final"] is not None]
if not A.quiet:
    print(f'{"":<8}{"Task":<26}{"Final":>7}{"blind":>8}{"ref":>7}{"Spread":>7}')
    for x in sorted(rows, key=lambda y: -(y["final"] if y["final"] is not None else -1)):
        if x.get("invalid_submission"):
            print(f'  {x["task"]} 0.000 invalid: {x["reason"]}')
            continue
        if x["final"] is None:
            print(f'  {x["task"]:<32} —      {x["why"]}')
            continue
        kind = "even" if "s3_even" in x["task"] else "golden"
        nm = x["task"].replace("/tmp/s3_even_", "").replace("/s_3", "")
        print(f'  {kind:<6}{nm:<24}{x["final"]:>7.3f}'
              f'{x["blind"]["score"]:>8.3f}{x["referenced"]["score"]:>7.3f}'
              f'{x["spread"]:>7.3f}')
if ok:
    import statistics as st
    g = [x["final"] for x in ok if "s3_even" not in x["task"]]
    e = [x["final"] for x in ok if "s3_even" in x["task"]]
    print()
    if g:
        print(f"  golden   median {st.median(g):.3f}  range {min(g):.2f}–{max(g):.2f}  n={len(g)}")
    if e:
        print(f"  equal-length control  median {st.median(e):.3f}  range {min(e):.2f}–{max(e):.2f}  n={len(e)}")
    if g and e:
        print(f"  separation  golden minimum {min(g):.3f}  control maximum {max(e):.3f}  "
              + ("No overlap" if min(g) > max(e) else "Warning: overlapping ranges"))

if len(ok) != len(rows):
    print(f"Incomplete scoring: {len(rows) - len(ok)}/{len(rows)} task(s) have no final score", file=sys.stderr)
    sys.exit(1)
