"""Teacher-selected categorical KL on K vocabulary entries plus a tail bucket.

All probabilities retain their FULL-vocabulary normalizer. The tail is computed
by logsumexp of the complement, never by subtracting nearly equal probabilities.
Only live assistant prediction positions are projected, in bounded chunks. The
custom backward recomputes logits chunkwise; no [trajectory, vocabulary] tensor
or autograd graph for every projection chunk is retained.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .answers import require

LOSS_NAME = "answer_opsd_topk"
OBJECTIVE = "teacher_topk_tail_reverse_kl"
OBJECTIVES = {"reverse": OBJECTIVE, "forward": "teacher_topk_tail_forward_kl"}


def kl_direction(config):
    matches = [direction for direction, objective in OBJECTIVES.items() if objective == config.get("loss")]
    require(len(matches) == 1, "Unknown top-K KL objective")
    direction = matches[0]
    require(config.get("kl_direction", direction) == direction, "KL direction and objective disagree")
    return direction


def selected_token_fallback_forbidden(*args, **kwargs):
    raise RuntimeError("Top-K OPSD requires its distribution worker/head; sampled-token fallback forbidden")


def _precision(tensor):
    return tensor if tensor.dtype == torch.float64 else tensor.float()


def _acc_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _project(hidden, weight):
    # Disable outer AMP for consistent forward/backward recomputation. The
    # operands already have the backbone's compute dtype (normally BF16).
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        z = _precision(F.linear(hidden, weight))
        return z - z.amax(-1, keepdim=True)


def bucket_logprobs(logits, ids):
    """K individually normalized classes + one aggregate complement class."""
    require(logits.ndim == ids.ndim == 2 and logits.shape[0] == ids.shape[0], "Top-K projection shape mismatch")
    require(1 <= ids.shape[1] < logits.shape[1], "Top-K must leave a nonempty tail")
    z = _precision(logits)
    z = z - z.amax(-1, keepdim=True)
    normalizer = z.logsumexp(-1, keepdim=True)
    top = z.gather(-1, ids) - normalizer
    complement = z.scatter(-1, ids, -torch.inf)
    tail = complement.logsumexp(-1, keepdim=True) - normalizer
    return torch.cat((top, tail), -1)


def categorical_kl(student, teacher, direction="reverse"):
    require(direction in ("reverse", "forward"), "Unknown KL direction")
    left, right = (student, teacher) if direction == "reverse" else (teacher, student)
    return (left.exp() * (left - right)).sum(-1)


@torch.no_grad()
def teacher_targets(hidden, weight, sampled_ids, k=64, chunk_size=128):
    require(type(k) is int and 1 < k < weight.shape[0], "Expected 1 < teacher top-K < vocabulary size")
    require(type(chunk_size) is int and chunk_size > 0, "Invalid projection chunk size")
    n = hidden.shape[0]
    ids = torch.empty((n, k), dtype=torch.long, device=hidden.device)
    bins = torch.empty((n, k + 1), dtype=_acc_dtype(hidden), device=hidden.device)
    action = bins.new_empty(n)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        z = _project(hidden[start:end], weight)
        selected = z.topk(k, dim=-1).indices
        ids[start:end] = selected
        bins[start:end] = bucket_logprobs(z, selected)
        action[start:end] = z.gather(-1, sampled_ids[start:end, None]).squeeze(-1) - z.logsumexp(-1)
    require(bool(torch.isfinite(bins).all() & torch.isfinite(action).all()), "Nonfinite teacher distribution")
    return ids.detach(), bins.detach(), action.detach()


class _ChunkedKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, ids, teacher, sampled_ids, chunk_size, direction):
        require(direction in ("reverse", "forward") and chunk_size > 0, "Invalid distribution objective")
        require(ids.shape[0] == teacher.shape[0] == hidden.shape[0] == sampled_ids.shape[0] and
                teacher.shape[1] == ids.shape[1] + 1, "Teacher target alignment changed")
        require(not teacher.requires_grad, "Teacher targets must be detached")
        require(bool(torch.isfinite(teacher).all()), "Nonfinite top-K teacher targets")
        require(bool((ids >= 0).all() & (ids < weight.shape[0]).all()), "Teacher vocabulary ID out of bounds")
        ordered = ids.sort(-1).values
        require(not bool((ordered[:, 1:] == ordered[:, :-1]).any()), "Duplicate teacher vocabulary IDs")
        require(bool(torch.allclose(teacher.logsumexp(-1), teacher.new_zeros(len(teacher)), atol=2e-5)),
                "Teacher top-K+tail probabilities do not sum to one")
        ctx.chunk_size, ctx.direction = chunk_size, direction
        ctx.save_for_backward(hidden, weight, ids, teacher)
        kl = hidden.new_empty(len(hidden), dtype=_acc_dtype(hidden))
        action, entropy, mass = torch.empty_like(kl), torch.empty_like(kl), torch.empty_like(kl)
        for start in range(0, len(hidden), chunk_size):
            end = min(len(hidden), start + chunk_size)
            z = _project(hidden[start:end], weight)
            lp = z - z.logsumexp(-1, keepdim=True)
            bins = bucket_logprobs(z, ids[start:end])
            kl[start:end] = categorical_kl(bins, teacher[start:end], direction)
            action[start:end] = lp.gather(-1, sampled_ids[start:end, None]).squeeze(-1)
            entropy[start:end] = -(lp.exp() * lp).sum(-1)
            mass[start:end] = bins[:, :-1].exp().sum(-1)
        require(bool(torch.isfinite(kl).all() & torch.isfinite(action).all()), "Nonfinite student distribution")
        ctx.mark_non_differentiable(action, entropy, mass)
        return kl, action, entropy, mass

    @staticmethod
    def backward(ctx, grad_kl, *_unused):
        hidden, weight, ids, teacher = ctx.saved_tensors
        # FP32 accumulation across chunks, cast only on return to the model.
        acc_dtype = _acc_dtype(hidden)
        dh = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight, dtype=acc_dtype) if ctx.needs_input_grad[1] else None
        for start in range(0, len(hidden), ctx.chunk_size):
            end = min(len(hidden), start + ctx.chunk_size)
            z = _project(hidden[start:end], weight)
            bins = bucket_logprobs(z, ids[start:end])
            p = (z - z.logsumexp(-1, keepdim=True)).exp()
            target = teacher[start:end]
            if ctx.direction == "reverse":
                gap = bins - target
                kl = (bins.exp() * gap).sum(-1, keepdim=True)
                dz = p * (gap[:, -1:] - kl)
                selected_grad = bins[:, :-1].exp() * (gap[:, :-1] - kl)
            else:
                complement = z.scatter(-1, ids[start:end], -torch.inf)
                conditional_tail = (complement - complement.logsumexp(-1, keepdim=True)).exp()
                dz = p - target[:, -1:].exp() * conditional_tail
                selected_grad = bins[:, :-1].exp() - target[:, :-1].exp()
            dz.scatter_(-1, ids[start:end], selected_grad)
            dz *= grad_kl[start:end, None]
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                if dh is not None:
                    dh[start:end] = (dz.to(weight.dtype) @ weight).to(hidden.dtype)
                if dw is not None:
                    dw.add_((dz.to(hidden.dtype).T @ hidden[start:end]).to(acc_dtype))
        return dh, None if dw is None else dw.to(weight.dtype), None, None, None, None, None


def distribution_kl(hidden, weight, ids, teacher, sampled_ids, chunk_size=128, direction="reverse"):
    return _ChunkedKL.apply(hidden, weight, ids, teacher.detach(), sampled_ids, chunk_size, direction)


def project_response(hidden, weight, sequences, num_actions, request):
    """Called inside the existing FSDP/VLM fused forward AFTER its backbone."""
    require(hidden.shape[:2] == sequences.shape and 0 < num_actions < sequences.shape[1],
            "Invalid response projection bounds")
    mask = request["mask"].bool()
    require(mask.shape == (len(sequences), num_actions), "Response loss mask alignment changed")
    # Hidden position t predicts token t+1. Keep all backbone context unchanged.
    selected = hidden[:, -num_actions-1:-1][mask]
    actions = sequences[:, -num_actions:][mask]
    if request["mode"] == "teacher":
        ids, bins, lp = teacher_targets(selected, weight, actions, request["k"], request["chunk_size"])
        return {"ids": ids, "teacher": bins, "logprobs": lp}
    require(request["mode"] in ("score", "train"), "Unknown top-K head mode")
    kl, lp, entropy, mass = distribution_kl(selected, weight, request["ids"], request["teacher"], actions,
                                          request["chunk_size"], request["direction"])
    return {"kl": kl, "logprobs": lp, "entropy": entropy, "student_mass": mass}
