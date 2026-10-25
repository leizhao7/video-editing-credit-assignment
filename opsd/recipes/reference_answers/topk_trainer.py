"""Sparse target transport and per-original-turn distribution diagnostics."""
from __future__ import annotations

import torch

from .answers import require


def expand_live(rows, key, mask):
    require(len(rows) == len(mask), "Worker output row count changed")
    dense = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    for i, row in enumerate(rows):
        value = torch.as_tensor(row[key], device=mask.device)
        require(value.shape == (int(mask[i].sum()),), "Worker sparse response alignment changed")
        dense[i, mask[i].bool()] = value.float()
    return dense


def teacher_forward(trainer, data, teacher_data):
    from skyrl.backends.skyrl_train.training_batch import TensorList
    cfg = trainer.opsd_plan["opsd"]
    teacher_data["loss_mask"] = data["loss_mask"]
    teacher_data.metadata["opsd_topk"] = cfg
    rows = trainer.dispatch.forward("ref", teacher_data).loss_fn_outputs
    require(len(rows) == len(data["loss_mask"]), "Teacher DP row count changed")
    for row, mask in zip(rows, data["loss_mask"]):
        require(row["ids"].shape == (int(mask.sum()), cfg["top_k"]) and
                row["teacher"].shape == (int(mask.sum()), cfg["top_k"] + 1), "Teacher top-K transport is incomplete")
    # Sparse int64 IDs and FP32 normalized log probabilities, no Python float lists.
    data["opsd_ids"] = TensorList([r["ids"].long() for r in rows])
    data["opsd_teacher"] = TensorList([r["teacher"].float().detach() for r in rows])
    data.metadata["opsd_topk"] = cfg
    return expand_live(rows, "logprobs", data["loss_mask"])


def student_forward(trainer, data):
    keys = [k for k in ("sequences", "attention_mask", "mm_token_type_ids", "response_mask", "loss_mask",
                        "pixel_values", "image_grid_thw", "opsd_ids", "opsd_teacher") if data.get(k) is not None]
    student = data.select(keys=keys, metadata_keys=["response_length", "opsd_topk"])
    rows = trainer.dispatch.forward("policy", student).loss_fn_outputs
    data["opsd_kl"] = expand_live(rows, "kl", data["loss_mask"])
    data["opsd_student_mass"] = expand_live(rows, "student_mass", data["loss_mask"])
    teacher_mass = [{"mass": lp[:, :-1].exp().sum(-1)} for lp in data["opsd_teacher"]]
    data["opsd_teacher_mass"] = expand_live(teacher_mass, "mass", data["loss_mask"])
    live = data["loss_mask"].bool()
    for metric, key in (("preupdate_topk_kl_mean", "opsd_kl"),
                        ("preupdate_teacher_topk_mass_mean", "opsd_teacher_mass"),
                        ("preupdate_student_topk_mass_mean", "opsd_student_mass")):
        trainer.all_metrics["opsd/" + metric] = data[key][live].mean().item()
    trainer.all_metrics["opsd/target_storage_bytes"] = sum(t.numel()*t.element_size()
        for key in ("opsd_ids", "opsd_teacher") for t in data[key])
    trainer.all_metrics["opsd/top_k"] = trainer.opsd_plan["opsd"]["top_k"]
    trainer.all_metrics["opsd/action_importance_weighting"] = 0.
    return expand_live(rows, "logprobs", data["loss_mask"])


def per_turn_audit(data, answers):
    records = []
    width = data.metadata["response_length"]
    for i, row in enumerate(data.metadata["ocrl_rows"]):
        offset = width - int(data["ocrl_packet_length"][i, 0])
        turns, covered = [], torch.zeros_like(data["loss_mask"][i], dtype=torch.bool)
        for span in row["causal_spans"]:
            lo, hi = offset + span["response_start"], offset + span["response_end"]
            require(0 <= lo < hi <= width and not bool(covered[lo:hi].any()), "Invalid original turn bounds")
            covered[lo:hi] = True
            live = data["loss_mask"][i, lo:hi].bool()
            kl = data["opsd_kl"][i, lo:hi][live]
            turn = {"request_id": span["request_id"], "turn": span["native_turn"],
                    "action_tokens": len(kl), "topk_kl_sum": kl.sum().item(),
                    "topk_kl_mean": kl.mean().item() if len(kl) else 0.}
            for name in ("teacher", "student"):
                values = data[f"opsd_{name}_mass"][i, lo:hi][live]
                turn[f"{name}_topk_mass_mean"] = values.mean().item() if len(values) else 0.
            turns.append(turn)
        require(not bool(data["loss_mask"][i][~covered].any()), "An assistant position has no original turn")
        total = sum(t["topk_kl_sum"] for t in turns)
        for turn in turns:
            turn["trajectory_kl_fraction"] = turn["topk_kl_sum"] / total if total > 1e-8 else None
        entry = answers.get(row["task_id"])
        records.append({"task": row["task_id"], "trajectory_id": row["trajectory_id"], "stage": entry["stage"],
                        "answer_sha256": entry["answer_sha256"], "outcome_reward": row["outcome"],
                        "outcome_loss_weight": 0., "topk_kl_sum": total,
                        "diagnostic_phase": "preupdate", "turns": turns})
    return records
