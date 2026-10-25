"""Opt-in FSDP worker paths, bypassing sampled-action policy gradient entirely.

Use SkyRL's existing DP dispatch, token microbatch scheduler, backward/optimizer,
and the audited fused VLM backbone. Targets travel as sparse TensorLists ordered
by original live response positions, not padded [batch, context, vocabulary].
"""
from contextlib import nullcontext

import torch

from .answers import require
from .topk import LOSS_NAME, OBJECTIVES, kl_direction


def _config(data):
    cfg = data.metadata.get("opsd_topk")
    require(cfg is not None and cfg["loss"] in OBJECTIVES.values() and cfg["gap_clip"] is None and
            cfg["topk_tail"] is True and cfg["action_importance_weighting"] is False,
            "Missing or incompatible distribution objective")
    kl_direction(cfg)
    return cfg


def _request(batch, cfg, mode):
    mask = batch["loss_mask"].bool()
    req = {"mask": mask, "mode": mode, "k": cfg["top_k"],
           "chunk_size": cfg["projection_chunk_size"], "direction": kl_direction(cfg)}
    if mode != "teacher":
        n = int(mask.sum())
        if batch.metadata.get("is_padding_batch"):
            require(n == 0, "Synchronization padding acquired live tokens")
            req["ids"] = torch.empty((0, cfg["top_k"]), dtype=torch.long, device=mask.device)
            req["teacher"] = torch.empty((0, cfg["top_k"] + 1), device=mask.device)
        else:
            require(batch.get("opsd_ids") is not None and batch.get("opsd_teacher") is not None,
                    "Distribution targets were lost before backward")
            selected_ids, selected_teacher = [], []
            for row, (ids, teacher) in enumerate(zip(batch["opsd_ids"], batch["opsd_teacher"])):
                count = int(mask[row].sum())
                if not count:  # DP pads may clone row-zero targets, but never learn them.
                    ids, teacher = ids[:0], teacher[:0]
                require(ids.shape == (count, cfg["top_k"]) and
                        teacher.shape == (ids.shape[0], cfg["top_k"] + 1), "Sparse teacher row/token alignment changed")
                selected_ids.append(ids)
                selected_teacher.append(teacher)
            req["ids"] = torch.cat(selected_ids).long()
            req["teacher"] = torch.cat(selected_teacher)
    return req


def _head(worker, batch, request):
    require(worker.cfg.algorithm.temperature == 1.0, "Top-K forward requires T=1")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if batch["sequences"].is_cuda else nullcontext()
    with amp:
        return worker.model(batch["sequences"], batch.metadata["response_length"],
            attention_mask=batch["attention_mask"], temperature=1., return_output=True,
            compute_entropy=False, entropy_requires_grad=False,
            pixel_values=batch.get("pixel_values"), image_grid_thw=batch.get("image_grid_thw"),
            mm_token_type_ids=batch["mm_token_type_ids"], opsd_request=request)


def _device(worker):
    # CPU override exists only for deterministic integration tests.
    return getattr(worker, "_opsd_test_device", None) or torch.cuda.current_device()


@torch.no_grad()
def forward(worker, data, mode):
    from skyrl.backends.skyrl_train.workers.worker_utils import get_microbatch_iterator
    from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
    cfg = _config(data)
    require(mode in ("teacher", "score"), "Invalid forward mode")
    worker.model.eval()
    data["opsd_local_row"] = torch.arange(data.batch_size, device=data["sequences"].device)[:, None]
    iterator = get_microbatch_iterator(data, micro_batch_size=worker.cfg.micro_forward_batch_size_per_gpu,
                                       max_tokens_per_microbatch=worker.cfg.max_tokens_per_microbatch)
    rows = [None] * data.batch_size
    for batch in iterator:
        batch.to(_device(worker))
        result = _head(worker, batch, _request(batch, cfg, mode))
        if batch.metadata.get("is_padding_batch"):
            continue
        counts = batch["loss_mask"].bool().sum(-1).tolist()
        local_ids = batch["opsd_local_row"].flatten().tolist()
        offset = 0
        for index, count in zip(local_ids, counts):
            require(rows[index] is None, "Duplicate microbatch row")
            # Ray serializes tensors directly; do not create millions of Python floats.
            rows[index] = {key: value[offset:offset+count].detach().cpu() for key, value in result.items()}
            offset += count
        require(offset == len(result["logprobs"]), "Lost sparse prediction positions")
    require(all(row is not None for row in rows), "Microbatch reordering lost a row")
    return WorkerOutput(loss_fn_outputs=rows, metrics={})


def forward_backward(worker, data, loss_fn=None, loss_fn_config=None, return_per_token_outputs=True):
    from skyrl.backends.skyrl_train.workers.worker_utils import get_microbatch_iterator
    from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
    cfg = _config(data)
    require(loss_fn in (None, LOSS_NAME) and not loss_fn_config and worker.cfg.algorithm.policy_loss_type == LOSS_NAME,
            "Top-K worker cannot silently change objectives")
    require(data.metadata.get("opsd_topk_normalized") is True, "Global live-token normalization missing")
    require(not worker.cfg.algorithm.use_kl_loss and not worker.cfg.algorithm.use_entropy_loss,
            "Unexpected auxiliary loss")
    worker.model.train()
    iterator = get_microbatch_iterator(data, micro_batch_size=worker.cfg.micro_train_batch_size_per_gpu,
                                       max_tokens_per_microbatch=worker.cfg.max_tokens_per_microbatch)
    totals = dict(kl=0., entropy=0., student_mass=0., teacher_mass=0., count=0., objective=0.)
    for batch in iterator:
        batch.to(_device(worker))
        request = _request(batch, cfg, "train")
        result = _head(worker, batch, request)
        weights = batch["advantages"][request["mask"]].detach()
        require(bool(torch.isfinite(weights).all() & (weights > 0).all()), "Invalid normalized token weights")
        loss = (result["kl"] * weights).sum()
        require(bool(torch.isfinite(loss)), "Nonfinite top-K KL loss")
        # FSDP averages gradients. Sum normalized per-token losses globally.
        worker.strategy.backward(loss * worker.mesh_rank.dp_size, worker.model, worker.optimizer)
        totals["objective"] += loss.detach().item()
        totals["count"] += len(weights)
        for name in ("kl", "entropy", "student_mass"):
            totals[name] += result[name].detach().sum().item()
        totals["teacher_mass"] += request["teacher"][:, :-1].exp().sum().item()
    totals = worker.strategy.all_reduce(totals, op="sum", group=worker.device_mesh.get_group("dp"))
    require(totals["count"] > 0, "No live assistant positions reached distribution backward")
    metrics = {"policy_loss": totals["objective"], "final_loss": totals["objective"],
               "policy_entropy": totals["entropy"] / totals["count"],
               "policy_lr": worker.scheduler.get_last_lr()[0],
               "opsd/topk_kl_mean": totals["kl"] / totals["count"],
               "opsd/teacher_topk_mass_mean": totals["teacher_mass"] / totals["count"],
               "opsd/student_topk_mass_mean": totals["student_mass"] / totals["count"],
               "opsd/action_tokens": totals["count"], "opsd/top_k": cfg["top_k"],
               "opsd/action_importance_weighting": 0., "opsd/forward_kl": float(kl_direction(cfg) == "forward"),
               "num_microbatches": float(len(iterator))}
    # Training's caller consumes metrics only. Keep row count without dense output.
    return WorkerOutput(loss_fn_outputs=[{} for _ in range(data.batch_size)], metrics=metrics)
