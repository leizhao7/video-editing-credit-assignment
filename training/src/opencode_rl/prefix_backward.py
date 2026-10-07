"""Share causal forward work while invoking SkyRL loss once per ORIGINAL turn.

No second policy-loss implementation. In particular geometric IS masks and KL
remain turn-local; original padded-row KL and global token GRPO denominators are
not replaced by packet counts. Metrics explicitly average original real turns.
"""
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

from .contract import ParityError

OVERLAY_SOURCE_SHA256 = 'cbb03d3f49a8f48b330cd3b01df1f6ba02436ab85fb4ee59bec437017d4c089f'


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


def _probe_gather(worker, value):
    """All ranks follow this control path, even when diagnostics are disabled."""
    import torch.distributed as dist
    size = worker.mesh_rank.dp_size
    if dist.is_initialized():
        group = worker.device_mesh.get_group('dp')
        if dist.get_world_size(group) != size:
            raise ParityError('Gradient probe DP group size differs from worker mesh')
        values = [None]*size
        dist.all_gather_object(values, value, group=group)
        return values
    if size != 1:
        raise ParityError('Gradient probe control is missing its distributed process group')
    return [value]


def _probe_plan(worker, data):
    """Synchronize failures BEFORE any rank can enter diagnostic collectives."""
    meta = data.metadata.get('prm_gradient_probe')
    checked, error = None, None
    try:
        if meta is not None:
            keys = {'version', 'enabled', 'global_step', 'interval', 'component_normalized',
                    'total_loss_tokens', 'audit_dir'}
            if (type(meta) is not dict or set(meta) != keys or meta['version'] != 1 or
                    type(meta['enabled']) is not bool or type(meta['global_step']) is not int or
                    meta['global_step'] < 0 or type(meta['interval']) is not int or meta['interval'] != 10 or
                    type(meta['component_normalized']) is not bool or
                    type(meta['total_loss_tokens']) is not int or meta['total_loss_tokens'] < 0 or
                    not isinstance(meta['audit_dir'], str) or not Path(meta['audit_dir']).is_absolute() or
                    Path(meta['audit_dir']) == Path('/')):
                raise ParityError('Invalid versioned gradient probe metadata')
            checked = dict(meta)
            if checked['enabled']:
                cfg = worker.cfg.algorithm
                if (not checked['component_normalized'] or checked['total_loss_tokens'] <= 0 or
                        cfg.policy_loss_type != 'rollout_is' or cfg.temperature != 1. or cfg.use_entropy_loss or
                        type(cfg.use_kl_loss) is not bool or
                        not math.isfinite(cfg.kl_loss_coef) or cfg.kl_loss_coef < 0 or
                        getattr(cfg, 'loss_reduction', 'token_mean') != 'token_mean' or
                        getattr(worker.cfg, 'update_epochs_per_batch', 1) != 1):
                    raise ParityError('Unqualified gradient probe objective or normalization')
            checked['already_sampled'] = getattr(worker, '_prm_gradient_probe_last_step', None) == checked['global_step']
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    gathered = _probe_gather(worker, {'plan': checked, 'error': error})
    if any(item['error'] for item in gathered):
        raise ParityError('Gradient probe metadata failed on a DP rank: ' +
                          '; '.join(item['error'] for item in gathered if item['error']))
    if any(item['plan'] != gathered[0]['plan'] for item in gathered):
        raise ParityError('Gradient probe plan or replay state differs across DP ranks')
    if (checked is None or not checked['enabled'] or checked['already_sampled'] or
            checked['global_step'] == 0 or
            (checked['global_step'] != 1 and checked['global_step'] % checked['interval'])):
        return None
    return checked


def _probe_components(batch, experience):
    """Match auxiliary global-normalized components after standard trim logic.

    Stage repadding copies row 0's auxiliary values, but the empty causal spans
    and zero loss prove that these are synchronization clones, not samples.
    Such rows get zero diagnostic components and never enter scope identities.
    """
    import torch
    mask, combined = experience.loss_mask, experience.advantages
    if (mask.ndim != 2 or combined is None or combined.shape != mask.shape or
            not torch.isfinite(combined).all() or mask.shape[1] < 1 or
            not bool(((mask == 0) | (mask == 1)).all())):
        raise ParityError('Invalid actual loss/advantage tensors for gradient probe')
    n, width = mask.shape
    spans = experience.ocrl_spans.tensors
    if len(spans) != n:
        raise ParityError('Gradient probe lost causal span rows')
    real = torch.tensor([span.numel() > 0 for span in spans], dtype=torch.bool, device=mask.device)
    if bool(mask[~real].any()):
        raise ParityError('Synchronization padding has an actual policy signal')
    if (experience.ocrl_packet_length.shape != (n, 1) or experience.ocrl_original_count.shape != (n, 1) or
            experience.ocrl_packet_length.dtype not in (torch.int32, torch.int64) or
            experience.ocrl_original_count.dtype not in (torch.int32, torch.int64) or
            bool((experience.ocrl_original_count <= 0).any())):
        raise ParityError('Malformed causal length/original denominator in gradient probe')
    for i in range(n):
        if not real[i]:
            continue
        boundaries, size = spans[i], int(experience.ocrl_packet_length[i, 0])
        active = experience.attention_mask[i].bool()
        count, previous = int(active.sum()), 0
        if (boundaries.ndim != 2 or boundaries.shape[1] != 2 or
                boundaries.dtype not in (torch.int32, torch.int64) or not 0 < size <= width or
                count-size < 1 or not bool(active[-count:].all()) or bool(active[:-count].any())):
            raise ParityError('Unqualified real packet bounds before gradient probe')
        covered = torch.zeros(width, dtype=torch.bool, device=mask.device)
        for lo, hi in boundaries.tolist():
            if not previous <= lo < hi <= size:
                raise ParityError('Overlapping or escaped gradient-probe action bounds')
            covered[width-size+lo:width-size+hi] = True
            previous = hi
        if boundaries[0, 0] != 0 or previous != size or bool(mask[i, ~covered].any()):
            raise ParityError('Gradient probe includes an unproven tool/context gap')
    components = {}
    for label, field in (('outcome', 'ocrl_outcome_advantages'), ('process', 'ocrl_process_advantages')):
        value = batch.get(field)
        if not bool(real.any()):
            components[label] = torch.zeros_like(combined)
            continue
        if (not torch.is_tensor(value) or not value.is_floating_point() or value.ndim != 2 or
                value.shape[0] != n or value.shape[1] < width):
            raise ParityError('Missing or truncated globally-normalized gradient component')
        value = value.to(device=combined.device)
        if (not bool(torch.isfinite(value[real]).all()) or
                bool(value[real, :value.shape[1]-width].any())):
            raise ParityError('Auxiliary gradient component lost a nonzero trimmed prefix')
        aligned = value[:, -width:].clone()
        aligned[~real] = 0.
        if bool(aligned[mask == 0].any()):
            raise ParityError('Context/tool gap entered a gradient component')
        components[label] = aligned
    if not torch.allclose(combined[real], (components['outcome']+components['process'])[real], rtol=2e-6, atol=1e-10):
        raise ParityError('Combined gradient advantage differs from its normalized components')
    indices = batch.get('ocrl_probe_row_index')
    if bool(real.any()) and (not torch.is_tensor(indices) or indices.shape != (n, 1) or
            indices.dtype not in (torch.int32, torch.int64)):
        raise ParityError('Gradient probe lost its original training-row indices')
    scope, seen = [], set()
    for i, is_real in enumerate(real.tolist()):
        if not is_real:
            continue
        index = int(indices[i, 0])
        if index < 0 or index in seen:
            raise ParityError('Repeated or invalid real gradient-probe row index')
        seen.add(index)
        scope.append({'row_index': index, 'causal_spans': spans[i].tolist(),
            'packet_response_tokens': int(experience.ocrl_packet_length[i, 0]),
            'loss_tokens': int(mask[i].sum()), 'loss_active': bool(mask[i].any())})
    return components, {'selected_rows': scope, 'real_packets': len(scope),
        'loss_active_packets': sum(item['loss_active'] for item in scope),
        'loss_tokens': int(mask.sum()), 'padding_packets': n-len(scope)}


def _probe_callback(worker, batch, components, *, to_experience, native_micro):
    """Fresh native model graph per component; combined uses the original worker."""
    def replay(component):
        experience = to_experience(batch)
        if component == 'combined':
            result = prefix_micro(worker, experience, native_micro=native_micro)
        else:
            if component not in ('outcome', 'process', 'kl'):
                raise ParityError('Unknown gradient probe source component')
            surrogate = SimpleNamespace(**vars(worker))
            surrogate.cfg = copy(worker.cfg)
            surrogate.cfg.algorithm = copy(worker.cfg.algorithm)
            if component in ('outcome', 'process'):
                surrogate.cfg.algorithm.use_kl_loss = False
                advantage = components[component]
            else:
                import torch
                advantage = torch.zeros_like(experience.advantages)
            experience = replace(experience, advantages=advantage)
            result = prefix_micro(surrogate, experience, native_micro=native_micro)
        # Returned native LPs are already CPU scalars. These temporary vectors
        # are freed after comparison; only scalar deltas enter the audit.
        values = [value for row in result[1] for value in row['logprobs']]
        if replay.response_length is None:
            replay.response_length = len(values)
        if len(values) != replay.response_length:
            raise ParityError('Gradient replay changed the native response inventory')
        replay.response_logprobs[component] = values
        if component == 'combined':
            replay.logprob_differences = {label: max((abs(a-b) for a, b in zip(other, values)), default=0.)
                                         for label, other in replay.response_logprobs.items()}
            replay.response_logprobs.clear()
        return result
    replay.response_length = None
    replay.response_logprobs = {}
    replay.logprob_differences = {}
    return replay


def _run_prefix_probe(worker, batch, plan, *, to_experience, native_micro):
    import torch.distributed as dist
    from prm_v2.gradient_probe_v5 import run_probe
    components = scope = receipt = normalizer_receipt = None
    error = None
    try:
        from prm_v2.gradient_source_guard_v5 import verify_worker
        receipt = verify_worker(worker)
        normalizer_receipt = batch.metadata.get('prm_gradient_normalizer_receipt')
        digest = normalizer_receipt.get('receipt_sha256') if type(normalizer_receipt) is dict else None
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(character not in '0123456789abcdef' for character in digest)):
            raise ParityError('Gradient probe lost its verified native normalizer source receipt')
        experience = to_experience(batch)
        components, scope = _probe_components(batch, experience)
        if worker.cfg.algorithm.use_kl_loss and experience.base_action_log_probs is None:
            raise ParityError('KL-enabled gradient probe lost reference logprobs')
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    gathered = _probe_gather(worker, {'scope': scope, 'source_receipt': receipt,
        'normalizer_receipt': normalizer_receipt, 'error': error})
    if any(item['error'] for item in gathered):
        raise ParityError('Gradient probe microbatch failed on a DP rank: ' +
                          '; '.join(item['error'] for item in gathered if item['error']))
    identities = [row['row_index'] for item in gathered for row in item['scope']['selected_rows']]
    if len(identities) != len(set(identities)):
        raise ParityError('Same real training packet selected on multiple DP ranks')
    if any(item['source_receipt'] != gathered[0]['source_receipt'] for item in gathered):
        raise ParityError('Gradient probe runtime sources differ across DP ranks')
    if any(item['normalizer_receipt'] != gathered[0]['normalizer_receipt'] for item in gathered):
        raise ParityError('Gradient probe native normalizer sources differ across DP ranks')
    if sum(item['scope']['loss_tokens'] for item in gathered) > plan['total_loss_tokens']:
        raise ParityError('Gradient probe scope exceeds its full-step loss denominator')
    group = worker.device_mesh.get_group('dp') if dist.is_initialized() else None
    callback = _probe_callback(worker, batch, components, to_experience=to_experience, native_micro=native_micro)
    result, metrics = run_probe(worker, callback, group=group)
    metrics = dict(metrics)
    metrics.update({f'prm_gradient/replay_logprob_max_abs_delta/{component}': value
                    for component, value in callback.logprob_differences.items()})
    metrics['prm_gradient/replay_logprob_max_abs_delta'] = max(callback.logprob_differences.values(), default=0.)
    metric_error = ('Invalid scalar metrics' if any(not key.startswith('prm_gradient/') or
        type(value) not in (int, float) or not math.isfinite(value) for key, value in metrics.items()) else None)
    metric_records = _probe_gather(worker, {'metrics': metrics if metric_error is None else {}, 'error': metric_error})
    metrics = _probe_reduce_metrics(metric_records)
    scopes = [item['scope'] for item in gathered]
    tokens = sum(item['loss_tokens'] for item in scopes)
    metrics.update({'prm_gradient/scope_first_microbatch': 1.,
        'prm_gradient/scope_is_random_sample': 0., 'prm_gradient/scope_is_full_step': 0.,
        'prm_gradient/sample_loss_tokens': tokens,
        'prm_gradient/full_step_loss_tokens': plan['total_loss_tokens'],
        'prm_gradient/sample_loss_token_fraction': tokens/plan['total_loss_tokens'],
        'prm_gradient/sample_real_packets': sum(item['real_packets'] for item in scopes),
        'prm_gradient/sample_loss_active_packets': sum(item['loss_active_packets'] for item in scopes),
        'prm_gradient/sample_padding_packets': sum(item['padding_packets'] for item in scopes)})
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    save_error = None
    if rank == 0:
        try:
            from .store import atomic_json
            path = Path(plan['audit_dir']) / f"step_{plan['global_step']:06d}.json"
            if path.exists():
                raise ParityError('Refusing to overwrite a gradient probe audit')
            atomic_json(path, {'version': 1, 'global_step': plan['global_step'],
                'scope': 'first synchronized microbatch; potentially length-biased; not a full-step or random estimate',
                'parameter_scope': 'all trainable parameters when sampled=1', 'plan': plan, 'metrics': metrics,
                'source_receipt': receipt,
                'normalizer_receipt': normalizer_receipt,
                'metric_aggregation': 'timings and replay LP deltas are DP maximum; shard sizes use explicitly named global sums and per-rank maxima; gradient Gram scalars are DP-reduced before reporting',
                'replay_logprob_delta_reference': 'normal combined forward',
                'diagnostic_scalars_per_dp_rank': [item['metrics'] for item in metric_records],
                'selected_rows_per_dp_rank': scopes})
        except Exception as exc:
            save_error = type(exc).__name__ + ': ' + str(exc)
    errors = _probe_gather(worker, {'error': save_error})
    if any(item['error'] for item in errors):
        raise ParityError('Gradient probe audit save failed: ' + '; '.join(item['error'] for item in errors if item['error']))
    worker._prm_gradient_probe_last_step = plan['global_step']
    return result, metrics


def _probe_reduce_metrics(records):
    """Merge rank-local diagnostics without averaging actual gradient norms."""
    if any(item['error'] for item in records):
        raise ParityError('Gradient probe returned invalid scalar metrics on a DP rank')
    keys = set(records[0]['metrics'])
    if any(set(item['metrics']) != keys for item in records):
        raise ParityError('Gradient probe scalar schemas differ across DP ranks')
    result = {}
    local_sizes = {'prm_gradient/local_param_numel': 'prm_gradient/sharded_param_numel',
                   'prm_gradient/cpu_grad_archive_bytes': 'prm_gradient/cpu_grad_archive_bytes'}
    for key in keys:
        values = [item['metrics'][key] for item in records]
        if key == 'prm_gradient/seconds' or key.endswith('_seconds') or key.startswith('prm_gradient/replay_logprob_'):
            result[key] = max(values)
        elif key in local_sizes:
            result[local_sizes[key]+'_global_sum'] = sum(values)
            result[local_sizes[key]+'_per_rank_max'] = max(values)
        elif any(not math.isclose(value, values[0], rel_tol=1e-10, abs_tol=1e-12) for value in values):
            raise ParityError('Global gradient metric differs across DP ranks: ' + key)
        else:
            result[key] = values[0]
    return result


def prefix_forward_backward(worker, data, loss_fn=None, loss_fn_config=None, return_per_token_outputs=True):
    import torch.distributed as dist
    from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase, WorkerOutput
    from skyrl.backends.skyrl_train.workers.worker_utils import BaseBatchIterator, get_microbatch_iterator
    if loss_fn not in (None, 'rollout_is') or loss_fn_config:
        raise ParityError('Causal backward cannot silently change the qualified objective')
    probe_plan = _probe_plan(worker, data)
    iterator = get_microbatch_iterator(data, micro_batch_size=worker.cfg.micro_train_batch_size_per_gpu,
                                     max_tokens_per_microbatch=worker.cfg.max_tokens_per_microbatch)
    records, outputs, gradient_metrics = [], [], {}
    for index, batch in enumerate(iterator):
        if index == 0 and probe_plan is not None:
            (local, packet_outputs), gradient_metrics = _run_prefix_probe(worker, batch, probe_plan,
                to_experience=BaseBatchIterator.batch_to_experience,
                native_micro=PolicyWorkerBase._forward_backward_micro)
        else:
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
    metrics.update(gradient_metrics)
    return WorkerOutput(loss_fn_outputs=outputs if return_per_token_outputs else [{} for _ in outputs], metrics=metrics)
