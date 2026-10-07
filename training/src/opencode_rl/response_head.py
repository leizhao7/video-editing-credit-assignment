"""Project only reply-predicting hidden states; never shorten model context.

The backbone and differentiable FSDP head-weight gather run before this helper.
Original-turn consumers use only the final num_actions prediction positions.
Entropy keeps the worker's full sequence shape, with explicit zero padding in
positions that no qualified loss or metric consumes. No full-shape fake
logprobs are returned. Context gradients still flow through the full backbone.
"""
from numbers import Integral

from .contract import ParityError

_REPORTED_MODES = set()


def response_suffix_linear_ce(hidden, weight, sequences, num_actions, temperature,
                              *, linear_ce, compute_entropy):
    import torch
    import torch.nn.functional as functional

    if (isinstance(num_actions, bool) or not isinstance(num_actions, Integral)
            or sequences.ndim != 2 or hidden.ndim != 3 or weight.ndim != 2
            or sequences.shape != hidden.shape[:2]
            or sequences.shape[0] < 1 or not 0 < num_actions < sequences.shape[1]
            or hidden.shape[-1] != weight.shape[-1] or weight.shape[0] < 1
            or sequences.dtype != torch.long or hidden.dtype != weight.dtype
            or not hidden.is_floating_point() or sequences.device != hidden.device
            or weight.device != hidden.device or temperature != 1.
            or type(compute_entropy) is not bool):
        raise ParityError('Unqualified response-only vocabulary projection')
    width = int(num_actions)
    # Hidden at position t predicts token t+1. Selecting the last width hidden
    # positions would be off by one and would train on the wrong conditioning.
    selected = hidden[:, -width - 1:-1, :].contiguous()
    targets = sequences[:, -width:].contiguous()
    lp, entropy = linear_ce(selected, weight, targets, float(temperature), 'none', None)
    shape = (sequences.shape[0], width)
    if (lp.numel() != shape[0] * width or entropy.numel() != lp.numel()
            or lp.device != hidden.device or entropy.device != hidden.device):
        raise ParityError('Fused response projection returned misaligned outputs')
    lp = lp.reshape(shape)
    full_entropy = None
    if compute_entropy:
        # Native worker and causal original-turn slices retain their positions.
        # F.pad is differentiable, including if entropy loss is enabled later.
        full_entropy = functional.pad(entropy.reshape(shape),
                                      (sequences.shape[1] - width - 1, 1))
    mode = (hidden.device.type, torch.is_grad_enabled(), compute_entropy)
    if mode not in _REPORTED_MODES:
        _REPORTED_MODES.add(mode)
        print(f'[ocrl_response_head] context_positions={sequences.numel()} '
              f'projected_positions={targets.numel()} grad={mode[1]} '
              f'entropy={compute_entropy} full_backbone_context=true', flush=True)
    return lp, full_entropy
