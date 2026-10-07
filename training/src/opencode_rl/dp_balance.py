"""Repartition intact causal packets before any ref/policy forward is staged.

Only the qualified single-minibatch, token-mean path is supported. No token,
image, response span, behavior probability or GRPO sibling is removed. Row zero
stays first so native initial padding and later stage_chunks repadding remain
identical. Per-prompt contiguous ranges cannot survive this permutation and are
explicitly disabled (unused by token_mean); task identity travels with each row.
"""
from .contract import ParityError


def balanced_order(lengths, dp_size, token_budget):
    if (not lengths or any(type(n) is not int or n < 1 for n in lengths)
            or type(dp_size) is not int or dp_size < 1
            or type(token_budget) is not int or token_budget < 1):
        raise ParityError('Invalid DP load-balancing dimensions')
    n = len(lengths)
    width = (n + dp_size - 1) // dp_size
    capacity = [max(0, min(width, n - rank * width)) for rank in range(dp_size)]
    # Include native loss-masked row-zero copies in the cost estimate, including
    # the small-batch case where an entire rank contains only padding.
    cost = lambda length: length * length + token_budget * length
    loads = [(width - count) * cost(lengths[0]) for count in capacity]
    groups = [[] for _ in capacity]
    groups[0].append(0)
    loads[0] += cost(lengths[0])
    for index in sorted(range(1, n), key=lambda i: (-lengths[i], i)):
        rank = min((k for k in range(dp_size) if len(groups[k]) < capacity[k]),
                   key=lambda k: (loads[k], len(groups[k]), k))
        groups[rank].append(index)
        loads[rank] += cost(lengths[index])
    order = [i for group in groups for i in group]
    if order[0] != 0 or sorted(order) != list(range(n)):
        raise ParityError('DP partition changed the packet inventory or padding seed')
    original = lengths + [lengths[0]] * (width * dp_size - n)
    original_peak = max(sum(cost(v) for v in original[i:i + width])
                        for i in range(0, len(original), width))
    if max(loads) >= original_peak:
        return list(range(n))
    return order


def rebalance_causal_batch(data, *, dp_size, token_budget, loss_reduction, tensor_list=None):
    """Permute all CPU row fields together, once, before any model sees them."""
    import torch
    if tensor_list is None:
        from skyrl.backends.skyrl_train.training_batch import TensorList
        tensor_list = TensorList
    meta = data.metadata
    rows = meta.get('ocrl_rows', [])
    if (loss_reduction != 'token_mean' or not rows
            or any('causal_spans' not in row for row in rows)
            or 'ocrl_dp_layout' in meta or 'critic_mini_batch_boundaries' in meta):
        raise ParityError('Unqualified or repeated causal DP repartition')
    n, total = len(rows), len(data['sequences'])
    boundaries = meta.get('policy_mini_batch_boundaries')
    if (type(dp_size) is not int or dp_size < 1
            or total != ((n + dp_size - 1) // dp_size) * dp_size
            or boundaries is None or [tuple(b) for b in boundaries] != [(0, n)]
            or meta.get('pad_size') != total - n):
        raise ParityError('DP balancing requires one complete policy minibatch and suffix padding')
    for key in ('uids', 'is_last_step'):
        if key in meta and (not isinstance(meta[key], (list, tuple)) or len(meta[key]) != total):
            raise ParityError(f'DP row metadata has an invalid length: {key}')
    for key, value in meta.items():
        if (key not in ('uids', 'is_last_step', 'ocrl_rows', 'policy_mini_batch_boundaries',
                        'policy_prompt_boundaries') and isinstance(value, (list, tuple)) and len(value) in (n, total)):
            raise ParityError(f'Unqualified additional DP row metadata: {key}')
    if 'uids' not in meta or any(uid != row['task_id'] for uid, row in zip(meta['uids'], rows)):
        raise ParityError('DP task identities were already misaligned')
    for key, value in data.items():
        if value is not None and (not isinstance(value, (torch.Tensor, tensor_list))
                or len(value) != total or value.device.type != 'cpu'):
            raise ParityError(f'DP field is not a complete CPU row array: {key}')
    counts = data['attention_mask'].sum(dim=1).tolist()
    if (any(counts[i] != max(s['end'] for s in row['causal_spans']) for i, row in enumerate(rows))
            or counts[n:] != [counts[0]] * (total - n)
            or bool(data['loss_mask'][n:].any())
            or any(span.numel() for span in data['ocrl_spans'].tensors[n:])):
        raise ParityError('Causal lengths or native zero-loss padding changed')
    order = balanced_order(counts[:n], dp_size, token_budget)
    indices = order + list(range(n, total))
    width = total // dp_size

    def loads(permutation):
        return [{'tokens': sum(counts[i] for i in permutation[start:start + width]),
                 'attention_proxy': sum(counts[i] ** 2 for i in permutation[start:start + width])}
                for start in range(0, total, width)]

    before, after = loads(list(range(total))), loads(indices)
    metrics = {'ocrl/dp_rebalanced': float(indices != list(range(total)))}
    for label, values in (('before', before), ('after', after)):
        for kind in ('tokens', 'attention_proxy'):
            sample = [value[kind] for value in values]
            metrics[f'ocrl/dp_{kind}_max_to_mean_{label}'] = max(sample) / (sum(sample) / dp_size)
    # Allocate one field at a time, releasing its old array immediately. The
    # original generator lists/journal are not mutated. No new tensor copies
    # are made for individual images inside TensorList.
    index = torch.tensor(indices, dtype=torch.long)
    if indices != list(range(total)):
        for key in data:
            value = data[key]
            if value is not None:
                data[key] = value[index]
    for key in ('uids', 'is_last_step'):
        if key in meta:
            meta[key] = [meta[key][i] for i in indices]
    meta['ocrl_rows'] = [rows[i] for i in order]
    meta['policy_prompt_boundaries'] = None
    meta['ocrl_dp_layout'] = {'format': 'ocrl-dp-balance-v1', 'dp_size': dp_size,
        'new_to_original_real_rows': order, 'before': before, 'after': after,
        'token_budget': token_budget, 'padding_rows': total - n}
    meta.setdefault('metrics', {}).update(metrics)
    return metrics
