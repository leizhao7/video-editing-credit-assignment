"""Share causal forward work while invoking SkyRL loss once per ORIGINAL turn.

No second policy-loss implementation. In particular geometric IS masks and KL
remain turn-local; original padded-row KL and global token GRPO denominators are
not replaced by packet counts. Metrics explicitly average original real turns.
"""
from contextlib import nullcontext
from dataclasses import replace
import math
from types import SimpleNamespace

from .contract import ParityError


class CollectedBackward:
    def __init__(self):
        self.losses = []

    def backward(self, loss, model, optimizer):
        self.losses.append(loss)


class RecordedForward:
    def __init__(self, logprobs, entropy, sequence):
        self.logprobs, self.entropy, self.sequence = logprobs, entropy, sequence

    def train(self):
        return self

    def __call__(self, sequence, num_actions, **kwargs):
        import torch
        if (not torch.equal(sequence, self.sequence) or num_actions != self.logprobs.shape[1] or
                kwargs['temperature'] != 1. or not kwargs['return_output'] or
                not kwargs['compute_entropy'] or kwargs['entropy_requires_grad']):
            raise ParityError('Native worker requested an unproven causal slice')
        return self.logprobs, {'entropy': self.entropy}


def prefix_micro(worker, experience, *, native_micro, device=None):
    import torch
    cfg = worker.cfg.algorithm
    if cfg.policy_loss_type != 'rollout_is' or cfg.temperature != 1. or cfg.use_entropy_loss:
        raise ParityError('Causal backward is qualified only for outcome rollout_is at T=1 without entropy loss')
    if (experience.ocrl_spans is None or experience.ocrl_packet_length is None or
            experience.ocrl_original_count is None or experience.sub_seq_lengths is not None or
            experience.rollout_expert_indices is not None):
        raise ParityError('Missing causal transport or unsupported routing/sequence packing')
    device = torch.cuda.current_device() if device is None else device
    experience.to_device(device)
    actor = worker.model
    actor.train()
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if experience.sequences.is_cuda else nullcontext()
    with amp:
        lp, output = actor(experience.sequences, experience.num_actions,
            attention_mask=experience.attention_mask, temperature=1., return_output=True,
            compute_entropy=True, entropy_requires_grad=False, pixel_values=experience.pixel_values,
            image_grid_thw=experience.image_grid_thw, mm_token_type_ids=experience.mm_token_type_ids)
    if lp.shape != experience.loss_mask.shape or not bool(torch.isfinite(lp).all()):
        raise ParityError('Causal forward returned invalid probabilities')
    entropy = output['entropy']
    if entropy.shape != experience.sequences.shape:
        raise ParityError('Causal forward entropy positions changed')
    collector, records, packet_outputs = CollectedBackward(), [], []
    fields = ('action_log_probs', 'base_action_log_probs', 'values', 'returns', 'advantages',
              'loss_mask', 'response_mask', 'rollout_logprobs', 'kl')
    for i, boundaries in enumerate(experience.ocrl_spans.tensors):
        size = int(experience.ocrl_packet_length[i, 0])
        denominator = int(experience.ocrl_original_count[i, 0])
        if denominator < 1 or denominator % worker.mesh_rank.dp_size:
            raise ParityError('Original padded KL denominator changed')
        if boundaries.numel() == 0:
            if bool(experience.loss_mask[i].any()):
                raise ParityError('Padding packet acquired policy loss')
            packet_outputs.append({'logprobs': []})
            continue
        active = experience.attention_mask[i].bool()
        count = int(active.sum())
        left, prompt_size = len(active)-count, count-size
        action_offset = experience.num_actions-size
        if (size < 1 or prompt_size < 1 or action_offset < 0 or
                not bool(active[left:].all()) or bool(active[:left].any()) or
                boundaries.ndim != 2 or boundaries.shape[1] != 2):
            raise ParityError('Packed sequence was truncated, repadded, or lost its action spans')
        covered = torch.zeros_like(experience.loss_mask[i], dtype=torch.bool)
        previous = 0
        for lo, hi in boundaries.tolist():
            if not previous <= lo < hi <= size:
                raise ParityError('Original action bounds overlap or escape packet')
            a, b = action_offset+lo, action_offset+hi
            end = left+prompt_size+hi
            seq = experience.sequences[i:i+1, left:end]
            covered[a:b] = True
            previous = hi
            if bool(experience.mm_token_type_ids[i, left+prompt_size+lo:end].any()):
                raise ParityError('Original generated actions became input media')
            values = {key: None if getattr(experience, key) is None else getattr(experience, key)[i:i+1, a:b]
                      for key in fields}
            original = replace(experience, **values, sequences=seq,
                attention_mask=torch.ones_like(seq), num_actions=hi-lo,
                mm_token_type_ids=experience.mm_token_type_ids[i:i+1, left:end],
                pixel_values=None, image_grid_thw=None, router_padding_mask=None,
                ocrl_spans=None, ocrl_packet_length=None, ocrl_original_count=None)
            surrogate = SimpleNamespace(**vars(worker))
            surrogate.model = RecordedForward(lp[i:i+1, a:b], entropy[i:i+1, left:end], seq)
            surrogate.strategy = collector
            record = native_micro(surrogate, original,
                microbatch_weight=worker.mesh_rank.dp_size/denominator, return_per_token_outputs=False)
            record.pop('loss_fn_outputs', None)
            # Native sample std is undefined for a one-token original turn.
            # This affects reporting only, never probabilities/masks/losses.
            record['ocrl/single_token_std_undefined'] = 0.
            std = record.get('loss_metrics/is_ratio_std')
            if hi-lo == 1 and std is not None and math.isnan(std):
                record['loss_metrics/is_ratio_std'] = 0.
                record['ocrl/single_token_std_undefined'] = 1.
            records.append(record)
        if previous != size or bool(experience.loss_mask[i][~covered].any()):
            raise ParityError('A context gap or trailing token entered training loss')
        packet_outputs.append({'logprobs': lp[i, action_offset:].detach().cpu().tolist()})
    # Even ranks with only synchronization padding must execute one backward
    # through the actual FSDP graph, keeping collective counts identical.
    loss = torch.stack(collector.losses).sum() if collector.losses else lp.sum()*0.
    if not bool(torch.isfinite(loss)):
        raise ParityError('Non-finite original-turn objective')
    worker.strategy.backward(loss, actor, worker.optimizer)
    return records, packet_outputs


def summarize_original_metrics(records):
    """Small mergeable scalar evidence; never gather raw tensors/trajectories."""
    keys = set().union(*(row.keys() for row in records)) if records else set()
    result = {}
    for key in keys:
        values = [row[key] for row in records]
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in values):
            raise ParityError(f'Invalid original-turn metric {key}')
        result[key] = {'sum': sum(values), 'min': min(values), 'max': max(values)}
    return {'count': len(records), 'metrics': result}


def combine_original_metrics(summaries):
    count = sum(item['count'] for item in summaries)
    if not count:
        raise ParityError('No original policy turns reached backward')
    live = [item for item in summaries if item['count']]
    keys = set(live[0]['metrics'])
    if any(set(item['metrics']) != keys for item in live):
        raise ParityError('Original-turn metric schemas differ across DP ranks')
    result = {}
    for key in keys:
        values = [item['metrics'][key] for item in live]
        if key.endswith('_min'):
            value = min(v['min'] for v in values)
        elif key.endswith('_max'):
            value = max(v['max'] for v in values)
        elif key == 'loss' or key.endswith('_loss'):
            value = sum(v['sum'] for v in values)
        else:
            value = sum(v['sum'] for v in values)/count
        result[key] = value
    result['ocrl/original_turns_backward'] = float(count)
    result['ocrl/metrics_are_original_turn_means'] = 1.
    return result


def prefix_forward_backward(worker, data, loss_fn=None, loss_fn_config=None, return_per_token_outputs=True):
    import torch.distributed as dist
    from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase, WorkerOutput
    from skyrl.backends.skyrl_train.workers.worker_utils import BaseBatchIterator, get_microbatch_iterator
    if loss_fn not in (None, 'rollout_is') or loss_fn_config:
        raise ParityError('Causal backward cannot silently change the qualified objective')
    iterator = get_microbatch_iterator(data, micro_batch_size=worker.cfg.micro_train_batch_size_per_gpu,
                                     max_tokens_per_microbatch=worker.cfg.max_tokens_per_microbatch)
    records, outputs = [], []
    for batch in iterator:
        local, packet_outputs = prefix_micro(worker, BaseBatchIterator.batch_to_experience(batch),
                                            native_micro=PolicyWorkerBase._forward_backward_micro)
        records.extend(local)
        outputs.extend(packet_outputs)
    summary = summarize_original_metrics(records)
    summary.update(microbatches=len(iterator), padding_microbatches=getattr(iterator, 'num_padding_microbatches', 0))
    summaries = [None]*worker.mesh_rank.dp_size
    if dist.is_initialized():
        dist.all_gather_object(summaries, summary, group=worker.device_mesh.get_group('dp'))
    elif len(summaries) == 1:
        summaries = [summary]
    else:
        raise ParityError('Causal backward is missing its distributed process group')
    metrics = combine_original_metrics(summaries)
    metrics['num_microbatches'] = sum(s['microbatches'] for s in summaries)/len(summaries)
    metrics['num_padding_microbatches'] = sum(s['padding_microbatches'] for s in summaries)/len(summaries)
    return WorkerOutput(loss_fn_outputs=outputs if return_per_token_outputs else [{} for _ in outputs], metrics=metrics)
