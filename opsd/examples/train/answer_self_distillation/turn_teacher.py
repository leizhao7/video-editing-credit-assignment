"""Per-turn suffix-conditioned ref forwards, mapped back to packed student rows.

Teacher: original prefix H_t + demo_t + original student action a_t.
Student: original packed trajectory, unchanged. No demo from another turn can
enter the ref sequence. Only original live assistant positions produce targets.
"""
import math
import time

import torch

from .answers import require


def columns(data, context):
    span = context.span
    offset = data.metadata["response_length"] - int(data["ocrl_packet_length"][context.row, 0])
    lo, hi = offset + span["response_start"], offset + span["response_end"]
    require(0 <= lo < hi <= data.metadata["response_length"] and
            hi - lo == span["end"] - span["start"], "Original turn response bounds changed")
    return lo, hi


def chunks(jobs, cfg):
    """Length-bucket ref requests; cap actual left-padded tokens including DP pads."""
    pending = []
    for job in sorted(jobs, key=lambda j: j["context"].span["end"] + len(j["hint_ids"])):
        length = job["context"].span["end"] + len(job["hint_ids"])
        require(4 * length <= cfg["teacher_dispatch_tokens"], "Teacher dispatch cannot fit one four-rank batch")
        padded = 4 * math.ceil((len(pending) + 1) / 4)
        if pending and (padded > cfg["teacher_dispatch_rows"] or padded * length > cfg["teacher_dispatch_tokens"]):
            yield pending
            pending = []
        pending.append(job)
    if pending:
        yield pending


def make_turn_batch(data, jobs, tokenizer, max_context, *, merge_size):
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch, TensorList
    require(jobs and data.get("rollout_expert_indices") is None, "Expected dense-model turn jobs")
    require(data.get("mm_token_type_ids") is not None, "Exact original multimodal type IDs required")
    width = max(j["context"].span["end"] + len(j["hint_ids"]) for j in jobs)
    response_width = max(j["context"].span["end"] - j["context"].span["start"] for j in jobs)
    require(response_width < width <= max_context, "Teacher context overflow; no truncation")
    count = 4 * math.ceil(len(jobs) / 4)
    ids = data["sequences"].new_full((count, width), tokenizer.pad_token_id)
    attention = data["attention_mask"].new_zeros((count, width))
    types = data["mm_token_type_ids"].new_zeros((count, width))
    loss = data["loss_mask"].new_zeros((count, response_width))
    response = data["response_mask"].new_zeros((count, response_width))
    pixels, grids = [], []
    for i in range(count):
        # All ranks participate even for a small final chunk; cloned pads never
        # contribute targets. The expensive backbone still uses its token cap.
        job = jobs[i] if i < len(jobs) else jobs[0]
        ctx, extra = job["context"], job["hint_ids"]
        r, start, end = ctx.row, ctx.span["start"], ctx.span["end"]
        original_mask = data["attention_mask"][r].bool()
        left = len(original_mask) - int(original_mask.sum())
        require(not original_mask[:left].any() and original_mask[left:].all(), "Only exact left padding supported")
        original = data["sequences"][r, left:]
        require(original.tolist() == ctx.sequence and 0 < start < end <= len(original), "Original context/targets changed")
        require(extra and not (set(extra) & set(tokenizer.all_special_ids)), "Invalid privileged suffix tokens")
        extra = original.new_tensor(extra)
        combined = torch.cat((original[:start], extra, original[start:end]))
        original_types = data["mm_token_type_ids"][r, left:]
        combined_types = torch.cat((original_types[:start], torch.zeros_like(extra), original_types[start:end]))
        ids[i, -len(combined):], attention[i, -len(combined):], types[i, -len(combined):] = combined, 1, combined_types
        lo, hi = columns(data, ctx)
        action_len = end - start
        require(torch.equal(original[start:end], data["sequences"][r, -data.metadata["response_length"]:][lo:hi]),
                "Teacher labels must be original student tokens, never demo tokens")
        # response_mask is structural, not a training weight. DP alignment rows
        # clone a real action and must retain its span for microbatch trimming.
        # Only loss_mask distinguishes real targets from synchronization pads.
        response[i, -action_len:] = data["response_mask"][r, lo:hi]
        require(bool(response[i].any()), "Original action has no structural response span")
        if i < len(jobs):
            loss[i, -action_len:] = data["loss_mask"][r, lo:hi]
        # Historical images are an exact prefix of the packed VLM tensors.
        n = len(ctx.features)
        grid = data["image_grid_thw"][r][:n]
        require(len(grid) == n and (not n or bool((grid > 0).all())), "Incomplete historical image grid")
        patches = sum(int(g.prod()) for g in grid)
        pixel = data["pixel_values"][r][:patches]
        require(len(pixel) == patches, "Incomplete historical image pixels")
        expected_types = torch.zeros_like(original_types[:end])
        for feature, g in zip(ctx.features, grid):
            offset, length = feature["offset"], feature["length"]
            require(feature["modality"] == "image" and 0 <= offset < offset + length <= start and
                    int(g.prod()) == length * merge_size ** 2 and
                    feature["data"]["pixel_values"]["shape"][0] == int(g.prod()), "Historical image geometry changed")
            require(not bool(expected_types[offset:offset+length].any()), "Overlapping historical images")
            expected_types[offset:offset+length] = 1
        require(torch.equal(expected_types, original_types[:end]), "Future/missing image entered teacher prefix")
        pixels.append(pixel)
        grids.append(grid)
    result = TrainingInputBatch({"sequences": ids, "attention_mask": attention, "mm_token_type_ids": types,
        "loss_mask": loss, "response_mask": response, "pixel_values": TensorList(pixels), "image_grid_thw": TensorList(grids)})
    result.metadata = {"response_length": response_width}
    return result


@torch.no_grad()
def teacher_forward(trainer, data):
    from skyrl.backends.skyrl_train.training_batch import TensorList
    cfg = trainer.opsd_plan["opsd"]
    started = time.monotonic()
    jobs = trainer.turn_demonstrations.prepare(data)
    trainer.all_metrics["opsd/demonstration_seconds"] = time.monotonic() - started
    started = time.monotonic()
    mask = data["loss_mask"].bool()
    covered = torch.zeros_like(mask)
    k = cfg["top_k"]
    # Targets are sparse in the ORIGINAL packed student order, not turn order
    # after length bucketing, nor Gemini action length/order.
    ranks = [m.long().cumsum(0) - 1 for m in mask]
    selected = [torch.empty((int(m.sum()), k), dtype=torch.long) for m in mask]
    teacher = [torch.empty((int(m.sum()), k + 1), dtype=torch.float32) for m in mask]
    dense = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    provenance, max_length, num_dispatch = {}, 0, 0
    for chunk in chunks(jobs, cfg["demonstrator"]):
        batch = make_turn_batch(data, chunk, trainer.tokenizer, cfg["teacher_max_context"],
                               merge_size=trainer.opsd_plan["contract"]["spatial_merge_size"])
        max_length = max(max_length, batch["sequences"].shape[1])
        batch.metadata["opsd_topk"] = cfg
        outputs = trainer.dispatch.forward("ref", batch).loss_fn_outputs
        require(len(outputs) == len(batch["loss_mask"]), "Per-turn ref DP row count changed")
        for i, output in enumerate(outputs):
            n = int(batch["loss_mask"][i].sum())
            require(output["ids"].shape == (n, k) and output["teacher"].shape == (n, k + 1) and
                    output["logprobs"].shape == (n,), "Per-turn sparse target alignment changed")
            require(bool(torch.isfinite(output["teacher"]).all() & torch.isfinite(output["logprobs"]).all()),
                    "Nonfinite per-turn teacher distribution")
            if i >= len(chunk):
                require(n == 0, "DP padding acquired teacher targets")
                continue
            ctx = chunk[i]["context"]
            lo, hi = columns(data, ctx)
            r = ctx.row
            require(not bool(covered[r, lo:hi].any()), "Duplicate per-turn teacher targets")
            covered[r, lo:hi] = True
            live = mask[r, lo:hi]
            indices = ranks[r][lo:hi][live].cpu()
            selected[r][indices] = output["ids"].detach().cpu().long()
            teacher[r][indices] = output["teacher"].detach().cpu().float()
            dense[r, lo:hi][live] = output["logprobs"].to(dense)
            key = ctx.span["request_id"]
            require(key not in provenance, "Repeated demo request identity")
            provenance[key] = chunk[i]["provenance"]
        num_dispatch += 1
        del batch, outputs
    require(not bool(mask[~covered].any()), "Original assistant tokens lack teacher targets")
    data["opsd_ids"], data["opsd_teacher"] = TensorList(selected), TensorList(teacher)
    data.metadata.update(opsd_topk=cfg, opsd_demonstrations=provenance)
    trainer.all_metrics.update({"opsd/teacher_forward_seconds": time.monotonic() - started,
        "opsd/teacher_max_sequence_tokens": max_length, "opsd/teacher_turn_dispatches": num_dispatch,
        "opsd/demonstration_turns": len(jobs), "opsd/hint_tokens_mean": sum(len(j["hint_ids"]) for j in jobs) / len(jobs)})
    return dense
