"""Combine the S4 blind and referenced judgements.

    final = 0.6 * blind + 0.4 * referenced

The blind judgement assesses the delivered audiovisual edit. The referenced
judgement uses the reference edit to assess what the available material supports,
including dialogue placement, music and audiovisual transitions.

Both components are required; a missing component is reported rather than
silently renormalising the remaining weight.

    python3 s4e_score.py --tasks path/to/tasks.json --suffix _example
"""
import argparse, json, os, sys

if __name__ != "__main__":
    raise ImportError(f"{__file__} is a command-line script that writes files")

from score_cli import add_task_arguments, validate_weights, task_paths, read_component, evaluate_tasks

P = argparse.ArgumentParser()
add_task_arguments(P)
P.add_argument("--suffix", default="", help="Read s4e_score<suffix>.json / s4e_score_ref<suffix>.json")
P.add_argument("--w-blind", type=float, default=0.6)
P.add_argument("--w-ref", type=float, default=0.4)
P.add_argument("--quiet", action="store_true")
A = P.parse_args()
validate_weights(P, (A.w_blind, A.w_ref))
if any(c in A.suffix for c in ("/", "\\", "\0")):
    P.error("suffix must not contain path separators or NUL")
tasks = task_paths(P, A)


def combine(task):
    from grading_contract import validate_task, InvalidSubmission, valid_score, judge_only, DIRECT_JUDGE_CONTRACT
    try:
        validate_task(task, "s_4e")
    except InvalidSubmission as exc:
        out = {"task": task, "final": 0.0, "invalid_submission": True, "reason": str(exc)}
        return out
    name = task.split("training_task/")[-1] if "training_task" in task else os.path.basename(task)
    bp = f"{task}/evidence/s4e_score{A.suffix}.json"
    rp = f"{task}/evidence/s4e_score_ref{A.suffix}.json"
    b = read_component(bp)
    r = read_component(rp)
    miss = [n for n, x in (("blind", b), ("referenced", r))
            if x is None or not valid_score(x.get("score")) or (x.get("partial") and not judge_only())]
    if miss:
        return {"task": name, "final": None, "missing": miss,
                "why": f"Missing {', '.join(miss)}; no aggregate is produced. "
                       "A single component cannot stand in for the final score."}
    if judge_only():
        VERSION = DIRECT_JUDGE_CONTRACT
    else:
        from grading_consistency import approved_components, ReviewError, VERSION
        try:
            reviewed = approved_components(task, "s_4e", A.suffix)
        except ReviewError as exc:
            out = {"task": name, "final": None, "missing": ["consistency"], "why": str(exc)}
            return out
        b, r = reviewed["blind"], reviewed["referenced"]
    weight_sum = A.w_blind + A.w_ref
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"S4E weights must sum to 1, got {weight_sum}")
    fin = A.w_blind * b["score"] + A.w_ref * r["score"]
    out = {"task": name, "final": round(fin, 3),
           "consistency_contract": VERSION,
           "judgement_mode": "direct-judge" if judge_only() else "jointly-reconciled",
           "initial_scores": {"blind": b.get("initial_score", b["score"]),
                              "referenced": r.get("initial_score", r["score"])},
           "blind": {"score": b["score"], "per_rubric": b.get("per_rubric"), "weight": A.w_blind},
           "referenced": {"score": r["score"], "per_rubric": r.get("per_rubric"), "weight": A.w_ref},
           "weight_sum": weight_sum,
           "spread": round(abs(b["score"] - r["score"]), 3),
           "note": ("Initial blind/reference judgments are used directly without an additional consistency review; "
                    "submission validation follows the active policy." if judge_only() else
                    "Initial judgments are retained. Blind/reference components are jointly reviewed against the "
                    "delivered video and audio; the reference is used only for calibration.")}
    return out


rows = evaluate_tasks(tasks, combine, lambda task: f"{task}/evidence/s4e_final{A.suffix}.json")
ok = [x for x in rows if x["final"] is not None]
if not A.quiet:
    print(f'  {"Task":<28}{"Final":>8}{"Blind":>8}{"Ref":>8}{"Spread":>7}')
    for x in sorted(rows, key=lambda y: -(y["final"] if y["final"] is not None else -1)):
        if x.get("invalid_submission"):
            print(f'  {x["task"]} 0.000 invalid: {x["reason"]}')
            continue
        if x["final"] is None:
            print(f'  {x["task"]:<28}    —    {x["why"]}')
            continue
        print(f'  {x["task"]:<28}{x["final"]:>8.3f}{x["blind"]["score"]:>8.3f}'
              f'{x["referenced"]["score"]:>8.3f}{x["spread"]:>7.3f}')
if ok:
    import statistics as st
    v = [x["final"] for x in ok]
    print(f'\n  Complete {len(ok)}/{len(rows)}   Median {st.median(v):.3f}   '
          f'Range {min(v):.2f}–{max(v):.2f}   Standard deviation {st.pstdev(v):.3f}')

if len(ok) != len(rows):
    print(f"Incomplete scoring: {len(rows) - len(ok)}/{len(rows)} task(s) have no final score", file=sys.stderr)
    sys.exit(1)
