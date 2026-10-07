"""Opt-in real parameter-gradient measurement on one synchronized microbatch.

Three *fresh native* forward/backwards isolate outcome, weighted process, and
KL; a fourth is the normal combined training pass whose gradients/state remain.
No optimizer step, clipping, autograd.grad, full-parameter all-gather, or reward
proxy is used. Caller owns low-frequency/first-microbatch scheduling. The three
diagnostic reduced local gradient shards live on CPU (potentially large).

Distributed support is intentionally narrow: one CUDA device per worker, one
1-D FSDP2 Shard mesh, no offload/replication/Partial gradients. Unsupported state
fails closed. A collective failure is a fatal training error, not recoverable
telemetry. Parameter version checks do not detect illicit .data writes bypassing
PyTorch version counters; callbacks must be the trusted native loss/backward.
"""
from dataclasses import dataclass
import math
import sys
import time

SAMPLING_SCOPE = 'sampled_first_microbatch'
DEFINITION = 'preclip_preAdam_reduced_parameter_gradients'
COMPONENTS = ('outcome', 'process', 'kl')
EPSILON = 1e-12
CHUNK_ELEMENTS = 1048576
RECONSTRUCTION_TOLERANCE = 1e-4  # Numerical diagnostic threshold, not GPU parity proof.
GRAM_CANCELLATION_ULPS = 64  # Guard subtraction of nearly equal float64 Gram entries.
PREFIX = 'prm_gradient/'


class GradientProbeError(RuntimeError):
    pass


@dataclass
class _Context:
    distributed: bool
    group: object = None
    device: object = None
    mesh: object = None
    ranks: tuple = ()


def _named_parameters(model):
    # Explicit dedup also supports wrappers whose iterator does not expose the
    # remove_duplicate keyword. Shared/tied weights enter the Gram only once.
    result, seen = [], set()
    for name, value in model.named_parameters():
        if id(value) not in seen:
            seen.add(id(value))
            result.append((name, value))
    return result


def _local(value):
    from torch.distributed.tensor import DTensor
    return value.to_local() if isinstance(value, DTensor) else value


def _context(worker, group):
    import torch
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        if group is not None:
            raise GradientProbeError('Process group supplied without distributed initialization')
        return _Context(False)
    if not torch.cuda.is_available():
        raise GradientProbeError('Distributed probe requires the CUDA FSDP path')
    if dist.get_world_size(group) == 1:
        # Initialized single-rank FSDP still follows the same placement guards.
        pass
    mesh = getattr(getattr(worker, 'strategy', None), 'device_mesh', None)
    if group is None and mesh is not None and getattr(mesh, 'ndim', None) == 1:
        group = mesh.get_group()
    if group is None:
        group = dist.group.WORLD
    return _Context(True, group, torch.device('cuda', torch.cuda.current_device()), mesh,
                    tuple(dist.get_process_group_ranks(group)))


def _consensus_error(error, ctx):
    import torch
    if ctx.distributed:
        import torch.distributed as dist
        flag = torch.tensor(int(error is not None), dtype=torch.int32, device=ctx.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=ctx.group)
        if flag.item():
            messages = [None] * len(ctx.ranks)
            dist.all_gather_object(messages, str(error)[:500] if error else '', group=ctx.group)
            detail = '; '.join(f'rank {rank}: {message}' for rank, message in zip(ctx.ranks, messages) if message)
            raise GradientProbeError('Synchronized probe rejected: ' + detail)
    elif error is not None:
        raise GradientProbeError(str(error)) from error


def _any_rank(flag, ctx):
    import torch
    if not ctx.distributed:
        return bool(flag)
    import torch.distributed as dist
    value = torch.tensor(int(flag), dtype=torch.int32, device=ctx.device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=ctx.group)
    return bool(value.item())


def _validate_layout(worker, parameters, ctx):
    import torch
    from torch.distributed.tensor import DTensor, Shard
    if not parameters:
        raise GradientProbeError('No trainable parameters to measure')
    if ctx.distributed:
        mesh = ctx.mesh
        if (mesh is None or mesh.ndim != 1 or mesh.device_type != 'cuda' or
                tuple(mesh.mesh.reshape(-1).tolist()) != ctx.ranks):
            raise GradientProbeError('Only one-dimensional exact FSDP shard group is supported')
        config = getattr(getattr(worker, 'strategy', None), 'fsdp_config', None)
        if config is None or getattr(config, 'cpu_offload', None) is not False:
            raise GradientProbeError('Explicit FSDP CPU offload=false required')
        if getattr(getattr(worker, 'mesh_rank', None), 'dp_size', None) != len(ctx.ranks):
            raise GradientProbeError('Worker DP size differs from the gradient shard group')
    devices = set()
    for name, parameter in parameters:
        if not parameter.is_floating_point() or parameter.is_sparse:
            raise GradientProbeError('Unsupported trainable parameter type: ' + name)
        local = _local(parameter)
        devices.add(local.device)
        if ctx.distributed:
            if (not isinstance(parameter, DTensor) or len(parameter.placements) != 1 or
                    not isinstance(parameter.placements[0], Shard) or
                    parameter.device_mesh.ndim != 1 or
                    tuple(parameter.device_mesh.mesh.reshape(-1).tolist()) != ctx.ranks or
                    local.device != ctx.device):
                raise GradientProbeError('Only CUDA FSDP Shard parameters are supported: ' + name)
        elif isinstance(parameter, DTensor):
            raise GradientProbeError('DTensor requires the explicit distributed placement path')
    if len(devices) != 1:
        raise GradientProbeError('One owned parameter device per worker is required')
    if ctx.device is None:
        ctx.device = next(iter(devices))


class _State:
    def __init__(self, model, parameters, ctx):
        import torch
        self.model, self.ctx = model, ctx
        self.parameter_identity = self._identities(parameters)
        self.cpu_rng = torch.get_rng_state().clone()
        # One owned CUDA device, not get_rng_state_all(), which can initialize
        # unrelated GPUs and allocate contexts inside a Ray worker.
        self.cuda_rng = (torch.cuda.get_rng_state(ctx.device).clone()
                         if ctx.device.type == 'cuda' else None)
        self.modules = []
        self.buffer_copies = {}
        for module in model.modules():
            mapping = dict(module._buffers)
            self.modules.append((module, module.training, mapping, set(module._non_persistent_buffers_set)))
            for value in mapping.values():
                if value is not None and id(value) not in self.buffer_copies:
                    if value.device.type == 'cuda' and value.device != ctx.device:
                        raise GradientProbeError('Registered buffer belongs to another CUDA device')
                    self.buffer_copies[id(value)] = (value, value.detach().clone())

    @staticmethod
    def _identities(parameters):
        return {name: (id(p), p._version, tuple(p.shape), p.dtype, p.requires_grad,
                       _local(p)._version) for name, p in parameters}

    def check_parameters(self):
        if self._identities(_named_parameters(self.model)) != self.parameter_identity:
            raise GradientProbeError('A callback changed parameter identity/version/shape/trainability')

    def restore(self):
        import torch
        with torch.no_grad():
            for original, saved in self.buffer_copies.values():
                if original.shape != saved.shape or original.dtype != saved.dtype or original.device != saved.device:
                    raise GradientProbeError('Registered buffer storage changed incompatibly')
                original.copy_(saved)
            for module, training, mapping, nonpersistent in self.modules:
                module._buffers.clear()
                module._buffers.update(mapping)
                module._non_persistent_buffers_set = set(nonpersistent)
                module.training = training
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state(self.cuda_rng, self.ctx.device)


def _clear(parameters):
    for _, parameter in parameters:
        parameter.grad = None


def _checked_grad(parameter, name, ctx):
    from torch.distributed.tensor import DTensor, Shard
    grad = parameter.grad
    if grad is None:
        return None
    if grad.is_sparse or not grad.is_floating_point() or grad.shape != parameter.shape:
        raise GradientProbeError('Invalid reduced parameter gradient: ' + name)
    if ctx.distributed and (not isinstance(grad, DTensor) or len(grad.placements) != 1 or
            not isinstance(grad.placements[0], Shard) or grad.placements != parameter.placements or
            tuple(grad.device_mesh.mesh.reshape(-1).tolist()) != ctx.ranks):
        raise GradientProbeError('Gradient is not a completed matching FSDP shard: ' + name)
    if not ctx.distributed and isinstance(grad, DTensor):
        raise GradientProbeError('Unexpected DTensor gradient')
    local = _local(grad).detach()
    if local.shape != _local(parameter).shape or local.device != ctx.device:
        raise GradientProbeError('Gradient local shard/offload layout changed: ' + name)
    return local


def _capture(parameters, ctx):
    if ctx.device.type == 'cuda':
        import torch
        torch.cuda.synchronize(ctx.device)
    result = []
    for name, parameter in parameters:
        value = _checked_grad(parameter, name, ctx)
        result.append(None if value is None else value.to(device='cpu', copy=True).reshape(-1))
    return result


def _gram_local(parameters, components, ctx):
    import torch
    if ctx.device.type == 'cuda':
        torch.cuda.synchronize(ctx.device)
    gram = torch.zeros((4, 4), dtype=torch.float64)
    error_squared = torch.zeros((), dtype=torch.float64)
    local_numel = 0
    nonfinite_component, nonfinite_combined = False, False
    for index, (name, parameter) in enumerate(parameters):
        combined = _checked_grad(parameter, name, ctx)
        combined = None if combined is None else combined.reshape(-1)
        count = _local(parameter).numel()
        local_numel += count
        for start in range(0, count, CHUNK_ELEMENTS):
            end = min(count, start + CHUNK_ELEMENTS)
            zero = torch.zeros(end-start, dtype=torch.float64)
            values = [zero if component[index] is None else component[index][start:end].to(dtype=torch.float64)
                      for component in components]
            values.append(zero if combined is None else combined[start:end].to(device='cpu', dtype=torch.float64))
            matrix = torch.stack(values)
            finite = torch.isfinite(matrix).all(dim=1)
            if not bool(finite.all()):
                nonfinite_component |= not bool(finite[:3].all())
                nonfinite_combined |= not bool(finite[3])
                # All backwards have already completed. Invalid telemetry
                # must not clear/replace the normal combined gradient or
                # preempt the native optimizer's nonfinite handling. Continue
                # layout checks but never publish a partial finite-only Gram.
                continue
            # Small result, not a large GEMM; dot products avoid invoking an
            # aggressive threaded 4xN BLAS matmul in every training worker.
            for i in range(4):
                for j in range(i, 4):
                    product = torch.dot(matrix[i], matrix[j])
                    gram[i, j] += product
                    if i != j:
                        gram[j, i] += product
            error_squared += (matrix[3] - matrix[:3].sum(dim=0)).square().sum()
    values = torch.cat((gram.reshape(-1), error_squared.reshape(1)))
    return values, local_numel, nonfinite_component, nonfinite_combined


def _invalid_metrics(*, nonfinite_component, nonfinite_combined):
    """No fake zero norms/percentages for an unmeasurable gradient."""
    result = dict(decomposition_valid=0, nonfinite_component=int(nonfinite_component),
                  nonfinite_combined=int(nonfinite_combined),
                  reconstruction_numerical_tolerance=RECONSTRUCTION_TOLERANCE)
    names = ('reconstruction_component_relative_error', 'reconstruction_relative_error',
             'policy_pair_norm_share_process', 'policy_pair_projection_share_process',
             'outcome_process_cosine', 'cancellation')
    names += tuple(kind + '_' + component for kind in ('norm_share', 'projection_share')
                   for component in COMPONENTS)
    result.update({name + '_valid': 0 for name in names})
    # Raw replay-strength ratios are not an additive attribution either. An
    # unmeasurable/nonfinite replay must not produce a fabricated percentage.
    result['replay_norm_ratios_are_additive_contributions'] = 0
    result['replay_policy_pair_norm_share_process_valid'] = 0
    result.update({'replay_norm_share_' + name + '_valid': 0 for name in COMPONENTS})
    return result


def _metrics(gram, error_squared):
    names = COMPONENTS + ('combined',)
    result = {f'gram/{a}_{b}': gram[i][j] for i, a in enumerate(names) for j, b in enumerate(names)}
    norms = [math.sqrt(max(0., gram[i][i])) for i in range(4)]
    if not all(math.isfinite(value) for row in gram for value in row) or not math.isfinite(error_squared):
        raise GradientProbeError('Nonfinite distributed gradient statistics')
    for name, norm in zip(names, norms):
        result[name + '_norm'] = norm
    denominator, total = sum(norms[:3]), norms[3]
    # Separately replayed native-gradient strengths remain observable even
    # when BF16 roundoff prevents exact additive reconstruction. These ratios
    # explicitly do NOT claim a decomposition of the actual combined gradient,
    # an error bound, or an Adam update share. Existing attribution guards stay.
    result['replay_norm_ratios_are_additive_contributions'] = 0
    for index, name in enumerate(COMPONENTS):
        valid = denominator > EPSILON
        result['replay_norm_share_' + name + '_valid'] = int(valid)
        if valid:
            result['replay_norm_share_' + name] = norms[index] / denominator
    replay_pair_denominator = norms[0] + norms[1]
    result['replay_policy_pair_norm_share_process_valid'] = int(replay_pair_denominator > EPSILON)
    if replay_pair_denominator > EPSILON:
        result['replay_policy_pair_norm_share_process'] = norms[1] / replay_pair_denominator
    residual = math.sqrt(max(0., error_squared))
    stable_denominator = denominator if denominator > EPSILON else total
    stable_valid = stable_denominator > EPSILON
    stable_relative = residual / stable_denominator if stable_valid else None
    decomposition_valid = (stable_relative <= RECONSTRUCTION_TOLERANCE if stable_valid else residual <= EPSILON)
    result['decomposition_valid'] = int(decomposition_valid)
    result['nonfinite_component'] = result['nonfinite_combined'] = 0
    result['reconstruction_numerical_tolerance'] = RECONSTRUCTION_TOLERANCE
    result['reconstruction_component_relative_error_valid'] = int(stable_valid)
    if stable_valid:
        result['reconstruction_component_relative_error'] = stable_relative
    pair_denominator = norms[0] + norms[1]
    pair_squared = max(0., gram[0][0] + gram[1][1] + 2. * gram[0][1])
    pair_roundoff_floor = (GRAM_CANCELLATION_ULPS * sys.float_info.epsilon *
                          (abs(gram[0][0]) + abs(gram[1][1]) + 2. * abs(gram[0][1])))
    pair_norm = math.sqrt(pair_squared)
    result['policy_pair_combined_norm'] = pair_norm
    pair_valid = decomposition_valid and pair_denominator > EPSILON
    result['policy_pair_norm_share_process_valid'] = int(pair_valid)
    if pair_valid:
        result['policy_pair_norm_share_process'] = norms[1] / pair_denominator
    # The pair direction is derived from gO+gP, not the actual combined
    # gradient. Near-cancelling Gram subtraction cannot resolve that direction.
    pair_valid = decomposition_valid and pair_norm > EPSILON and pair_squared > pair_roundoff_floor
    result['policy_pair_projection_share_process_valid'] = int(pair_valid)
    if pair_valid:
        result['policy_pair_projection_share_process'] = (gram[1][0] + gram[1][1]) / pair_squared
    for index, name in enumerate(COMPONENTS):
        valid = decomposition_valid and denominator > EPSILON
        result['norm_share_' + name + '_valid'] = int(valid)
        if valid:
            result['norm_share_' + name] = norms[index] / denominator
        # Component-relative reconstruction can be excellent while a almost
        # cancelled total direction is wrong. Its projection needs BOTH guards.
        valid = (decomposition_valid and total > EPSILON and
                 residual / total <= RECONSTRUCTION_TOLERANCE)
        result['projection_share_' + name + '_valid'] = int(valid)
        if valid:
            result['projection_share_' + name] = gram[index][3] / gram[3][3]
    # This is the raw angle of the two replay gradients, even if they failed
    # to reconstruct the separately replayed combined gradient.
    cosine_valid = norms[0] > EPSILON and norms[1] > EPSILON
    result['outcome_process_cosine_valid'] = int(cosine_valid)
    if cosine_valid:
        result['outcome_process_cosine'] = max(-1., min(1., gram[0][1] / (norms[0]*norms[1])))
    result['cancellation_valid'] = int(decomposition_valid and denominator > EPSILON)
    if decomposition_valid and denominator > EPSILON:
        # 0 aligned; near 1 strong cancellation. No clamping conceals a
        # failed additive reconstruction (which can yield a negative value).
        result['cancellation'] = 1. - total / denominator
    result['reconstruction_absolute_error'] = residual
    result['reconstruction_relative_error_valid'] = int(total > EPSILON)
    if total > EPSILON:
        result['reconstruction_relative_error'] = result['reconstruction_absolute_error'] / total
    return result


def run_probe(worker, callback, *, group=None):
    """Return (normal combined callback result, flat numeric diagnostics).

    Callback must perform a fresh native forward/backward and no optimizer,
    scheduler, clipping, scaler, or manual FSDP sync changes. 'process' includes
    its actual training coefficient; 'kl' must be graph-connected even when 0.
    All ranks call in identical order, including ranks with padding-only rows.
    Existing accumulated gradients cause an all-rank skip and combined-only
    execution; probe gradients can never overwrite an accumulation in progress.
    """
    started = time.monotonic()
    if not callable(callback):
        raise GradientProbeError('Native replay callback required')
    ctx = _context(worker, group)
    all_parameters = _named_parameters(worker.model)
    parameters = [(name, parameter) for name, parameter in all_parameters if parameter.requires_grad]
    error = None
    try:
        _validate_layout(worker, parameters, ctx)
    except Exception as exc:
        error = exc
    _consensus_error(error, ctx)
    if _any_rank(any(parameter.grad is not None for _, parameter in parameters), ctx):
        combined_started = time.monotonic()
        value = callback('combined')
        combined_seconds = time.monotonic()-combined_started
        total_seconds = time.monotonic()-started
        return value, {PREFIX+'sampled': 0, PREFIX+'skipped_existing_grad': 1,
                       PREFIX+'seconds': total_seconds, PREFIX+'combined_native_seconds': combined_seconds,
                       PREFIX+'diagnostic_extra_seconds': max(0., total_seconds-combined_seconds)}
    state, error = None, None
    try:
        state = _State(worker.model, all_parameters, ctx)
    except Exception as exc:
        error = exc
    _consensus_error(error, ctx)
    components, combined_succeeded, phase_seconds = [], False, {}
    try:
        for name in COMPONENTS:
            phase_started, error = time.monotonic(), None
            try:
                state.restore()
                _clear(parameters)
            except Exception as exc:
                error = exc
            _consensus_error(error, ctx)
            callback(name)
            error, captured = None, None
            try:
                state.check_parameters()
                captured = _capture(parameters, ctx)
            except Exception as exc:
                error = exc
            _consensus_error(error, ctx)
            components.append(captured)
            _clear(parameters)
            phase_seconds[name+'_replay_seconds'] = time.monotonic()-phase_started
        error = None
        try:
            state.restore()
        except Exception as exc:
            error = exc
        _consensus_error(error, ctx)
        combined_started = time.monotonic()
        value = callback('combined')
        combined_seconds = time.monotonic()-combined_started
        combined_succeeded = True
        error, values, local_numel = None, None, None
        nonfinite_component, nonfinite_combined = False, False
        try:
            state.check_parameters()
            values, local_numel, nonfinite_component, nonfinite_combined = _gram_local(parameters, components, ctx)
        except Exception as exc:
            error = exc
        # No rank enters the scalar SUM while another rank rejected its local
        # layout checks. A failure within native backward is fatal and
        # is deliberately not treated as a safely recoverable metric failure.
        _consensus_error(error, ctx)
        nonfinite_component = _any_rank(nonfinite_component, ctx)
        nonfinite_combined = _any_rank(nonfinite_combined, ctx)
        if nonfinite_component or nonfinite_combined:
            metrics = _invalid_metrics(nonfinite_component=nonfinite_component,
                                       nonfinite_combined=nonfinite_combined)
        else:
            if ctx.distributed:
                import torch.distributed as dist
                values = values.to(ctx.device)
                dist.all_reduce(values, op=dist.ReduceOp.SUM, group=ctx.group)
                values = values.cpu()
            gram, error_squared = values[:16].reshape(4, 4).tolist(), float(values[16])
            metrics = _metrics(gram, error_squared)
        total_seconds = time.monotonic()-started
        metrics.update(sampled=1, skipped_existing_grad=0, param_count=len(parameters),
            param_numel=sum(parameter.numel() for _, parameter in parameters),
            local_param_numel=local_numel, diagnostic_cpu_shards=3,
            cpu_grad_archive_bytes=sum(value.numel()*value.element_size()
                for component in components for value in component if value is not None),
            native_passes=4, sampled_first_microbatch=1, preclip=1, preAdam=1,
            seconds=total_seconds, combined_native_seconds=combined_seconds,
            diagnostic_extra_seconds=max(0., total_seconds-combined_seconds), **phase_seconds)
        return value, {PREFIX+name: value for name, value in metrics.items()}
    finally:
        components.clear()
        if not combined_succeeded:
            _clear(parameters)
            state.restore()
