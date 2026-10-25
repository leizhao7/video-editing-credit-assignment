"""Trim left padding per micro-batch without changing any real trajectory token.

SkyRL builds one dense ``[batch, max_sequence_length]`` tensor before data-
parallel dispatch.  With VLMs, ``remove_microbatch_padding`` cannot be enabled,
so the stock sample iterator keeps that *global* width in every micro-batch.
One 72k-token trajectory consequently makes all micro-batches execute at 72k.

This patch is deliberately narrower than sequence packing:

* sort each DP rank's rows by real length so equally expensive collectives line
  up across ranks;
* remove only columns that are padding for every row in the current micro-batch;
* keep each row, token, image, mask, reward, and advantage otherwise unchanged;
* right-pad inference outputs back to the original response width and restore
  the original row order before they leave the worker.

Real tokens stay in the same order and retain the same position ids, because
SkyRL uses left padding and derives positions from the attention mask.  This is
therefore compatible with the VLM path that cannot use packed sequences.
"""
from __future__ import annotations

import os
from typing import Iterable

import torch


_APPLIED = False
_PRINTED = False

# Fields whose second dimension follows the full prompt+response sequence.
_SEQUENCE_FIELDS = frozenset(
    {
        "sequences",
        "attention_mask",
        "rollout_expert_indices",
        "router_padding_mask",
        "mm_token_type_ids",
    }
)

# Fields whose second dimension follows the response-only suffix.  These are
# right-aligned by convert_prompts_responses_to_batch_tensors().
_RESPONSE_FIELDS = frozenset(
    {
        "response_mask",
        "rewards",
        "loss_mask",
        "rollout_logprobs",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
    }
)


def _select_rows(data, indices: list[int]):
    """Index every batch-aligned field while leaving metadata for local repair."""
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

    selected = {}
    index_tensor = torch.tensor(indices, dtype=torch.long, device="cpu")
    for key, value in data.items():
        selected[key] = None if value is None else value[index_tensor]
    batch = TrainingInputBatch(selected)
    batch.metadata = dict(data.metadata or {})
    return batch


def _trim_common_left_padding(batch):
    """Return a view/copy with only all-row left padding columns removed."""
    attention_mask = batch.get("attention_mask")
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError("micro-batch trim requires a 2-D attention_mask")

    seq_width = int(attention_mask.shape[1])
    real_lengths = attention_mask.sum(dim=1, dtype=torch.long)
    local_seq_width = int(real_lengths.max().item())
    if local_seq_width <= 0:
        raise ValueError("micro-batch contains no real sequence token")

    response_mask = batch.get("response_mask")
    original_response_width = int((batch.metadata or {}).get("response_length", 0))
    if response_mask is not None:
        if response_mask.ndim != 2:
            raise ValueError("micro-batch trim requires a 2-D response_mask")
        local_response_width = int(
            response_mask.sum(dim=1, dtype=torch.long).max().item()
        )
    else:
        local_response_width = original_response_width
    if local_response_width <= 0:
        raise ValueError("micro-batch contains no real response token")
    if local_response_width > original_response_width:
        raise ValueError(
            "local response width exceeds batch metadata: "
            f"{local_response_width} > {original_response_width}"
        )

    seq_start = seq_width - local_seq_width
    response_start = original_response_width - local_response_width
    for key, value in list(batch.items()):
        if value is None:
            continue
        if key in _SEQUENCE_FIELDS:
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                raise TypeError(f"sequence field {key!r} is not a rank>=2 tensor")
            if int(value.shape[1]) != seq_width:
                raise ValueError(
                    f"sequence field {key!r} has width {value.shape[1]}, expected {seq_width}"
                )
            batch[key] = value[:, seq_start:]
        elif key in _RESPONSE_FIELDS:
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                raise TypeError(f"response field {key!r} is not a rank>=2 tensor")
            if int(value.shape[1]) != original_response_width:
                raise ValueError(
                    f"response field {key!r} has width {value.shape[1]}, "
                    f"expected {original_response_width}"
                )
            batch[key] = value[:, response_start:]

    batch.metadata = dict(batch.metadata or {})
    batch.metadata["response_length"] = local_response_width
    batch.metadata["s12_original_response_length"] = original_response_width
    batch.metadata["s12_trimmed_sequence_width"] = local_seq_width
    return batch


def _microbatch_indices(iterator) -> list[list[int]]:
    """Longest-first fixed-size chunks, stable for equal lengths.

    Every DP rank performs the same number of micro-steps.  Sorting each local
    shard longest-first aligns its length quantiles with the other ranks, so a
    different rank does not become the straggler at every successive FSDP
    collective.
    """
    cached = getattr(iterator, "_s12_trim_indices", None)
    if cached is not None:
        return cached
    lengths = iterator.data["attention_mask"].sum(dim=1, dtype=torch.long).tolist()
    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))
    size = int(iterator.sample_batch_size)
    chunks = [order[i : i + size] for i in range(0, len(order), size)]
    iterator._s12_trim_indices = chunks
    return chunks


def _iter_trimmed(iterator) -> Iterable:
    global _PRINTED
    chunks = _microbatch_indices(iterator)
    for indices in chunks:
        batch = _trim_common_left_padding(_select_rows(iterator.data, indices))
        if not _PRINTED:
            _PRINTED = True
            before = int(iterator.data["sequences"].shape[1])
            after = int(batch["sequences"].shape[1])
            print(
                f"[microbatch_trim] active rank-local={len(iterator.data)} "
                f"micro={iterator.sample_batch_size} first_width={before}->{after}",
                flush=True,
            )
        yield batch


def _restore_indexed_outputs(iterator, batches, chunks, padding_batches=0):
    """Restore row order/right padding for variable-width microbatch outputs."""
    real_batch_count = len(batches) - int(padding_batches)
    real_batches = batches[:real_batch_count]
    if not real_batches:
        raise ValueError("cannot restore an empty micro-batch output list")
    if len(chunks) != len(real_batches):
        raise ValueError(
            f"output count mismatch: {len(real_batches)} != {len(chunks)}"
        )

    n_rows = len(iterator.data)
    target_width = int((iterator.data.metadata or {})["response_length"])
    restored = {}
    for key, reference in real_batches[0].items():
        if reference is None:
            restored[key] = None
            continue
        if not isinstance(reference, torch.Tensor):
            raise TypeError(f"unsupported forward output {key!r}: {type(reference)}")
        if reference.ndim < 2:
            output = torch.empty(
                (n_rows, *reference.shape[1:]),
                dtype=reference.dtype,
                device=reference.device,
            )
        else:
            output = torch.zeros(
                (n_rows, target_width, *reference.shape[2:]),
                dtype=reference.dtype,
                device=reference.device,
            )

        for indices, micro_output in zip(chunks, real_batches):
            value = micro_output[key]
            if len(value) != len(indices):
                raise ValueError(
                    f"row count mismatch for {key!r}: {len(value)} != {len(indices)}"
                )
            row_index = torch.tensor(indices, dtype=torch.long, device=value.device)
            if value.ndim < 2:
                output[row_index] = value
            else:
                width = int(value.shape[1])
                if width > target_width:
                    raise ValueError(
                        f"output {key!r} width {width} exceeds original {target_width}"
                    )
                output[row_index, target_width - width :] = value
        restored[key] = output

    output_batch = type(real_batches[0])(restored)
    output_batch.metadata = iterator.data.metadata
    return output_batch


def _restore_outputs(iterator, batches):
    """Restore sample-based rank-local row order and output width."""
    return _restore_indexed_outputs(
        iterator, batches, _microbatch_indices(iterator)
    )


def _token_microbatch(iterator, indices):
    """Create and immediately trim one real token-packed microbatch."""
    batch = iterator._s12_stock_create_microbatch(indices)
    return _trim_common_left_padding(batch)


def _split_real_bins(groups, token_counts, target):
    groups = [list(group) for group in groups]
    while len(groups) < target:
        candidates = [
            index for index, group in enumerate(groups) if len(group) > 1
        ]
        if not candidates:
            raise RuntimeError(
                f"cannot split {len(groups)} real token bins to target {target}"
            )

        def savings(index):
            lengths = [token_counts[row] for row in groups[index]]
            return (max(lengths) * (len(lengths) - 1), max(lengths), len(lengths))

        selected = max(candidates, key=savings)
        ordered = sorted(
            groups[selected],
            key=lambda row: (-int(token_counts[row]), int(row)),
        )
        groups[selected : selected + 1] = [[ordered[0]], ordered[1:]]
    return groups


def _token_iterator_init(iterator, *args, **kwargs):
    """Equalize DP micro-step counts by splitting real bins, never fake rows.

    SkyRL's stock token iterator appends a one-row batch whose ``loss_mask`` is
    entirely zero.  PPO still computes rollout-logprob diagnostics on that row
    and calls ``max()`` on the resulting empty tensor.  Every rank has the same
    number of real samples, so the synchronized target can always be reached by
    subdividing a packed bin; this keeps every token and loss weight real while
    preserving collective alignment.
    """
    iterator._s12_stock_init(*args, **kwargs)
    target = len(iterator._microbatches) + iterator._num_padding_microbatches
    iterator._microbatches = _split_real_bins(
        iterator._microbatches, iterator._token_counts, target
    )
    iterator._num_padding_microbatches = 0


def _token_padding_microbatch(iterator):
    """Make a one-token, zero-loss collective-alignment microbatch."""
    batch = iterator._s12_stock_create_padding_microbatch()
    attention = batch["attention_mask"]
    attention.zero_()
    attention[:, -1] = 1
    response = batch["response_mask"]
    response.zero_()
    response[:, -1] = 1
    return _trim_common_left_padding(batch)


def _restore_token_outputs(iterator, batches):
    """Drop collective padding batches and restore token-packed row order."""
    return _restore_indexed_outputs(
        iterator,
        batches,
        iterator._microbatches,
        iterator._num_padding_microbatches,
    )


def apply() -> bool:
    """Patch SkyRL's fixed-size sample iterator once per Python process."""
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_TRIM_MICROBATCH", "0") != "1":
        return False
    from skyrl.backends.skyrl_train.workers.worker_utils import (
        SampleBasedBatchIterator,
        TokenBasedBatchIterator,
    )

    SampleBasedBatchIterator._s12_stock_iter = SampleBasedBatchIterator.__iter__
    SampleBasedBatchIterator._s12_stock_restore = (
        SampleBasedBatchIterator.reorder_and_combine_batches
    )
    SampleBasedBatchIterator.__iter__ = _iter_trimmed
    SampleBasedBatchIterator.reorder_and_combine_batches = _restore_outputs
    TokenBasedBatchIterator._s12_stock_create_microbatch = (
        TokenBasedBatchIterator._create_microbatch_from_indices
    )
    TokenBasedBatchIterator._s12_stock_init = TokenBasedBatchIterator.__init__
    TokenBasedBatchIterator._s12_stock_create_padding_microbatch = (
        TokenBasedBatchIterator._create_padding_microbatch
    )
    TokenBasedBatchIterator._s12_stock_restore = (
        TokenBasedBatchIterator.reorder_and_combine_batches
    )
    TokenBasedBatchIterator.__init__ = _token_iterator_init
    TokenBasedBatchIterator._create_microbatch_from_indices = _token_microbatch
    TokenBasedBatchIterator._create_padding_microbatch = (
        _token_padding_microbatch
    )
    TokenBasedBatchIterator.reorder_and_combine_batches = (
        _restore_token_outputs
    )
    _APPLIED = True
    print(
        "[microbatch_trim] sample- and token-based iterators patched",
        flush=True,
    )
    return True


def self_test() -> None:
    """CPU-only contract test used by the Modal config check."""
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
    from skyrl.backends.skyrl_train.workers.worker_utils import (
        SampleBasedBatchIterator,
        TokenBasedBatchIterator,
    )

    old = os.environ.get("SKYRL_TRIM_MICROBATCH")
    os.environ["SKYRL_TRIM_MICROBATCH"] = "1"
    try:
        apply()
        lengths = [3, 10, 6, 8]
        responses = [2, 6, 3, 4]
        width, response_width = 10, 6
        attention = torch.zeros((4, width), dtype=torch.long)
        response_mask = torch.zeros((4, response_width), dtype=torch.long)
        sequences = torch.zeros((4, width), dtype=torch.long)
        loss_mask = torch.zeros((4, response_width), dtype=torch.float32)
        for row, (seq_len, response_len) in enumerate(zip(lengths, responses)):
            attention[row, -seq_len:] = 1
            sequences[row, -seq_len:] = row + 1
            response_mask[row, -response_len:] = 1
            loss_mask[row, -response_len:] = row + 1
        data = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention,
                "response_mask": response_mask,
                "loss_mask": loss_mask,
            }
        )
        data.metadata = {"response_length": response_width}
        iterator = SampleBasedBatchIterator(data, sample_batch_size=2)
        micros = list(iterator)
        assert iterator._s12_trim_indices == [[1, 3], [2, 0]]
        assert [tuple(x["sequences"].shape) for x in micros] == [(2, 10), (2, 6)]
        assert [x.metadata["response_length"] for x in micros] == [6, 3]
        outputs = []
        for indices, micro in zip(iterator._s12_trim_indices, micros):
            local_width = micro.metadata["response_length"]
            value = torch.zeros((len(indices), local_width), dtype=torch.float32)
            for local_row, original_row in enumerate(indices):
                value[local_row] = original_row + 1
            from skyrl.backends.skyrl_train.training_batch import TrainingOutputBatch

            item = TrainingOutputBatch({"output": value})
            item.metadata = micro.metadata
            outputs.append(item)
        restored = iterator.reorder_and_combine_batches(outputs)["output"]
        assert tuple(restored.shape) == (4, response_width)
        for row in range(4):
            local_width = max(responses[i] for i in iterator._s12_trim_indices[0])
            if row in iterator._s12_trim_indices[1]:
                local_width = max(responses[i] for i in iterator._s12_trim_indices[1])
            assert torch.all(restored[row, -local_width:] == row + 1)
            assert torch.all(restored[row, : response_width - local_width] == 0)

        # Token packing is the production path when
        # trainer.max_tokens_per_microbatch > 0.  It must trim the global dense
        # width too; otherwise four short rows beside one 75k outlier still
        # execute as 4x75k and can exhaust an H200.
        token_iterator = TokenBasedBatchIterator(
            data, max_tokens_per_microbatch=18
        )
        token_micros = list(token_iterator)
        for indices, micro in zip(token_iterator._microbatches, token_micros):
            expected_width = max(lengths[index] for index in indices)
            assert int(micro["sequences"].shape[1]) == expected_width
            assert sum(lengths[index] for index in indices) <= 18
        token_outputs = []
        for indices, micro in zip(token_iterator._microbatches, token_micros):
            width = int(micro.metadata["response_length"])
            value = torch.zeros((len(indices), width), dtype=torch.float32)
            for local_row, original_row in enumerate(indices):
                value[local_row] = original_row + 1
            from skyrl.backends.skyrl_train.training_batch import TrainingOutputBatch

            item = TrainingOutputBatch({"output": value})
            item.metadata = micro.metadata
            token_outputs.append(item)
        token_restored = token_iterator.reorder_and_combine_batches(
            token_outputs
        )["output"]
        assert tuple(token_restored.shape) == (4, response_width)
        for row in range(4):
            assert torch.all(
                token_restored[row, -responses[row] :] == row + 1
            )

        # Exercise the distributed count-alignment padding path without a
        # process group: its dense sequence and response widths must both be 1.
        token_iterator._num_padding_microbatches = 1
        padded = list(token_iterator)[-1]
        assert tuple(padded["sequences"].shape) == (1, 1)
        assert tuple(padded["loss_mask"].shape) == (1, 1)
        assert int(padded["attention_mask"].sum().item()) == 1
        assert int(padded["loss_mask"].sum().item()) == 0

        split = _split_real_bins([[0, 1, 2], [3]], [10, 5, 4, 3], 4)
        assert len(split) == 4
        assert sorted(row for group in split for row in group) == [0, 1, 2, 3]
        assert all(group for group in split)
    finally:
        if old is None:
            os.environ.pop("SKYRL_TRIM_MICROBATCH", None)
        else:
            os.environ["SKYRL_TRIM_MICROBATCH"] = old
