"""Causal span provenance transported with actual DP-sliced training tensors."""
import math

from .contract import ParityError
from .training import original_request_rows


def verify_generated_packet(generated, index):
    row = generated['ocrl_rows'][index]
    spans, originals = row.get('causal_spans', []), row.get('original_rows', [])
    response = generated['response_ids'][index]
    mask, lp = generated['loss_masks'][index], generated['rollout_logprobs'][index]
    prompt = generated['prompt_token_ids'][index]
    mm = generated['mm_token_type_ids'][index][len(prompt):]
    if (row.get('causal_format') != 'ocrl-causal-prefix-v1' or not spans or
            len(spans) != len(originals) or row.get('original_turn_count') != len(originals) or
            row.get('packed_response_length') != len(response) or
            len(mask) != len(response) or len(lp) != len(response) or len(mm) != len(response)):
        raise ParityError('Missing or truncated causal packet provenance')
    previous, covered, expected_mask, seen = 0, [False]*len(response), [0]*len(response), set()
    for span, original in zip(spans, originals):
        lo, hi = span['response_start'], span['response_end']
        if (type(lo) is not int or type(hi) is not int or not previous <= lo < hi <= len(response) or
                span['start'] != len(prompt)+lo or span['end'] != len(prompt)+hi or
                original['response_length'] != hi-lo or len(span['original_loss_mask']) != hi-lo or
                any(v not in (0, 1) for v in span['original_loss_mask']) or any(mm[lo:hi])):
            raise ParityError('Original actions cannot overlap, move, or become input media')
        for key in ('request_id', 'input_sha256', 'weight_version', 'native_turn'):
            if span[key] != original[key]:
                raise ParityError('Original action identity changed in transport')
        for key in ('task_id', 'trajectory_id', 'contract_sha256', 'outcome'):
            if original[key] != row[key]:
                raise ParityError('Causal packet crossed task/trajectory/reward boundaries')
        if original['process'] != 0 or original['process_valid'] or original['request_id'] in seen:
            raise ParityError('Unsupported PRM or repeated original actions in packet')
        seen.add(original['request_id'])
        expected_mask[lo:hi] = span['original_loss_mask']
        covered[lo:hi] = [True]*(hi-lo)
        previous = hi
    # Full-group zero-variance filtering may zero an entire row, never add targets.
    if mask != expected_mask and any(mask):
        raise ParityError('Causal packet changed original loss masks')
    if spans[0]['response_start'] != 0 or previous != len(response):
        raise ParityError('Unaccounted leading or trailing packet response')
    if any(not live and lp[i] != 0. for i, live in enumerate(covered)):
        raise ParityError('Tool/context gap has an invented behavior probability')


def attach_prefix_transport(data, generated, *, dp_size, tensor_list=None):
    import torch
    rows = generated['ocrl_rows']
    if not any('causal_spans' in row for row in rows):
        return
    if not all('causal_spans' in row for row in rows) or type(dp_size) is not int or dp_size < 1:
        raise ParityError('Mixed packet modes or invalid original DP denominator')
    if tensor_list is None:
        from skyrl.backends.skyrl_train.training_batch import TensorList
        tensor_list = TensorList
    spans, lengths = [], []
    inventory = original_request_rows(rows)
    if len({row['request_id'] for row in inventory}) != len(inventory):
        raise ParityError('Original request repeated across packets')
    for i, row in enumerate(rows):
        verify_generated_packet(generated, i)
        spans.append(torch.tensor([[s['response_start'], s['response_end']] for s in row['causal_spans']], dtype=torch.long))
        lengths.append([len(generated['response_ids'][i])])
    size = len(data['sequences'])
    if size != math.ceil(len(rows)/dp_size)*dp_size:
        raise ParityError('Unexpected DP padding size for causal packets')
    for i in range(len(rows), size):
        if bool(data['loss_mask'][i].any()):
            raise ParityError('Distributed padding must have no actions')
        spans.append(torch.empty((0, 2), dtype=torch.long))
        lengths.append(lengths[0])
    data['ocrl_spans'] = tensor_list(spans)
    data['ocrl_packet_length'] = torch.tensor(lengths, dtype=torch.long)
    # Original row padding is retained in KL's denominator even though packed
    # DP clones carry no original action spans and contribute zero gradients.
    count = math.ceil(len(inventory)/dp_size)*dp_size
    data['ocrl_original_count'] = torch.full((size, 1), count, dtype=torch.long)
