"""Insert private text ONLY in the teacher's existing system message.

Never render/re-tokenize the student conversation. Preserve its entire token and
image suffix, including every tool observation and generated action. The same
packed response offsets therefore identify the same tokens on both forwards.
"""
from __future__ import annotations

import torch

from .answers import require


def encode_hint(tokenizer, hint, max_hint_tokens):
    ids = tokenizer.encode(hint, add_special_tokens=False)
    require(ids and len(ids) <= max_hint_tokens, "Teacher hint exceeds reserved budget; never truncate answers")
    require(not (set(ids) & set(tokenizer.all_special_ids)), "Private text injected a special/control token")
    return ids


def make_teacher_batch(data, hint_ids, tokenizer, max_context):
    """Return a distinct forward-only batch, plus scalar size diagnostics.

    Both inputs are left padded. Response tensors have width `response_length`,
    NOT sequence width. Short rows may have prompt positions within that suffix;
    their original zero loss mask still excludes those positions. Added hint text
    is before the FIRST response, so active action positions cannot move within
    the right-aligned response suffix. No sequence is clipped to make it fit.
    """
    require(data.get("rollout_expert_indices") is None, "MoE replay is not qualified for this recipe")
    sequences, attention = data["sequences"], data["attention_mask"]
    require(sequences.ndim == 2 and attention.shape == sequences.shape, "Invalid student tensor shape")
    require(len(hint_ids) == len(sequences), "Teacher hint row order/DP padding mismatch")
    require(data.get("mm_token_type_ids") is not None, "Exact multimodal type IDs required")
    response_length = data.metadata["response_length"]
    require(data["loss_mask"].shape == (len(sequences), response_length), "Response mask width differs")
    header = tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    require(header and end_id in tokenizer.all_special_ids, "Unsupported chat template: expected Qwen system message")
    teacher_ids, teacher_types, lengths = [], [], []
    for i, (sequence, mask, extra) in enumerate(zip(sequences, attention, hint_ids)):
        count = int(mask.sum())
        left = len(mask) - count
        require(count > 0 and not bool(mask[:left].any()) and bool(mask[left:].all()), "Only exact left padding supported")
        active = sequence[left:]
        require(active[:len(header)].tolist() == header, "Cannot identify original system header safely")
        ends = (active == end_id).nonzero().flatten()
        require(len(ends) > 0, "System message has no end marker")
        at = int(ends[0])
        packet_length = int(data["ocrl_packet_length"][i, 0])
        # Synchronization-only rows retain row zero's packet length, but their loss mask is zero.
        require(at < count - packet_length, "Teacher hint would be inserted inside the response")
        extra = torch.tensor(extra, dtype=sequence.dtype, device=sequence.device)
        require(count + len(extra) <= max_context, "Teacher context overflow; cannot truncate student history")
        combined = torch.cat((active[:at], extra, active[at:]))
        types = data["mm_token_type_ids"][i, left:]
        combined_types = torch.cat((types[:at], torch.zeros_like(extra), types[at:]))
        # This assertion checks exactly the positions whose logprobs enter loss.
        live = data["loss_mask"][i].bool()
        student_suffix = sequence[-response_length:]
        padded = torch.nn.functional.pad(combined, (max(0, response_length-len(combined)), 0), value=tokenizer.pad_token_id)
        require(torch.equal(padded[-response_length:][live], student_suffix[live]), "Teacher/student target alignment broke")
        teacher_ids.append(combined)
        teacher_types.append(combined_types)
        lengths.append(len(combined))
    width = max(max(lengths), response_length + 1)
    require(width <= max_context, "Padded teacher context exceeds supported budget")
    result = data.select(keys=[k for k in ("pixel_values", "image_grid_thw", "response_mask")
                             if data.get(k) is not None], metadata_keys=["response_length"])
    ids = sequences.new_full((len(sequences), width), tokenizer.pad_token_id)
    types = data["mm_token_type_ids"].new_zeros(ids.shape)
    masks = attention.new_zeros(ids.shape)
    for i, (row, row_types) in enumerate(zip(teacher_ids, teacher_types)):
        ids[i, -len(row):], types[i, -len(row):], masks[i, -len(row):] = row, row_types, 1
    result["sequences"], result["attention_mask"], result["mm_token_type_ids"] = ids, masks, types
    return result, {"opsd/teacher_max_sequence_tokens": max(lengths),
                    "opsd/hint_tokens_mean": sum(map(len, hint_ids)) / len(hint_ids)}
