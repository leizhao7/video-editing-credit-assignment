"""Sampled reverse-KL guidance following SkyRL's OPD example, not full-vocab KL.

A = stop_gradient(log teacher(y | x, answer, history) - log student(y | x, history)).
The existing rollout_is policy loss applies current/behavior importance ratios.
There is no group baseline, outcome loss, PRM, or second reference KL penalty.
"""
from __future__ import annotations

import torch

from .answers import require


def answer_opsd_passthrough(token_level_rewards, **kwargs):
    """Stable import identity across `python -m` and Ray worker processes."""
    return token_level_rewards, token_level_rewards


@torch.no_grad()
def sampled_advantages(teacher, student, mask, gap_clip=5.0):
    require(teacher.shape == student.shape == mask.shape, "Distillation tensor shape mismatch")
    require(gap_clip is None or (type(gap_clip) in (int, float) and 0 < gap_clip < float("inf")), "Invalid gap clip")
    require(bool(((mask == 0) | (mask == 1)).all()), "Only binary assistant loss masks supported")
    live = mask.bool()
    require(bool(live.any()), "No student action tokens remain")
    require(bool(torch.isfinite(teacher[live]).all() & torch.isfinite(student[live]).all()),
            "Nonfinite teacher/student probabilities on actual actions")
    raw = torch.zeros_like(student, dtype=torch.float32)
    raw[live] = teacher[live].float() - student[live].float()
    advantage = raw if gap_clip is None else raw.clamp(-gap_clip, gap_clip)
    values = raw[live]
    metrics = {"opsd/logprob_gap_mean": values.mean().item(), "opsd/logprob_gap_abs_mean": values.abs().mean().item(),
               "opsd/advantage_abs_mean": advantage[live].abs().mean().item(),
               "opsd/positive_token_fraction": (values > 0).float().mean().item(),
               "opsd/gap_clip_fraction": 0.0 if gap_clip is None else (values.abs() > gap_clip).float().mean().item(),
               "opsd/action_tokens": int(live.sum()), "opsd/outcome_loss_weight": 0.0, "opsd/process_loss_weight": 0.0}
    return advantage.detach(), metrics


def per_turn_audit(data, raw_teacher, student, advantages, answers):
    """Compact persisted signal assignment for each original action/trajectory.

    This is a logprob-guidance contribution, NOT a claim to measure gradient norm.
    Turns are identified by immutable request IDs, never matched across tasks by turn number.
    """
    records = []
    response_length = data.metadata["response_length"]
    for i, row in enumerate(data.metadata["ocrl_rows"]):
        offset = response_length - int(data["ocrl_packet_length"][i, 0])
        turns = []
        for span in row["causal_spans"]:
            lo, hi = offset + span["response_start"], offset + span["response_end"]
            mask = data["loss_mask"][i, lo:hi].bool()
            values = advantages[i, lo:hi][mask]
            raw = (raw_teacher[i, lo:hi][mask] - student[i, lo:hi][mask]).float()
            turns.append({"request_id": span["request_id"], "turn": span["native_turn"],
                          "action_tokens": int(mask.sum()), "guidance_abs_sum": values.abs().sum().item(),
                          "guidance_mean": values.mean().item() if len(values) else 0.0,
                          "unclipped_gap_mean": raw.mean().item() if len(raw) else 0.0})
        total = sum(x["guidance_abs_sum"] for x in turns)
        for turn in turns:
            turn["trajectory_guidance_fraction"] = turn["guidance_abs_sum"] / total if total else 0.0
        entry = answers.get(row["task_id"])
        records.append({"task": row["task_id"], "trajectory_id": row["trajectory_id"], "stage": entry["stage"],
                        "answer_sha256": entry["answer_sha256"], "outcome_reward": row["outcome"],
                        "outcome_loss_weight": 0.0, "guidance_abs_sum": total, "turns": turns})
    return records
