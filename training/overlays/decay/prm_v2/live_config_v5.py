"""Explicit online opt-in; importing/configuring never makes a provider call."""
import math
import re


DEFAULTS = dict(protocol='v5', weight=0.5, window=10, votes=3,
    concurrency=64, max_pending_windows=256, request_timeout_s=60.,
    window_timeout_s=150., finish_timeout_s=180., requests_per_second=4.,
    model='gemini-3.8-flash', max_attempts=2)

# Separate from the judge protocol: these are training-only, low-frequency
# parameter-gradient diagnostics, not extra provider requests.
GRADIENT_DEFAULTS = dict(version=1, interval=10, first_step=True,
    scope='first_synchronized_microbatch_all_trainable_parameters',
    components=['outcome', 'weighted_process', 'kl'],
    numerics='native_flash_and_torch_deterministic_v1')

TRAINING_NUMERICS_ENV = {
    'FLASH_ATTENTION_DETERMINISTIC': '1',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'PRM_V5_TRAINING_NUMERICS': GRADIENT_DEFAULTS['numerics'],
}


def training_numerics_env(plan):
    """Training subprocess only; never add these flags to inference/base_env."""
    return {} if gradient_config_for_plan(plan) is None else dict(TRAINING_NUMERICS_ENV)


def _require_training_numerics_environment():
    import os
    if any(os.environ.get(key) != value for key, value in TRAINING_NUMERICS_ENV.items()):
        raise ValueError('PRM deterministic kernel environment missing or changed')


def _require_training_numerics_state():
    import torch
    _require_training_numerics_environment()
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise ValueError('PRM policy worker requires strict native deterministic kernels')


def deterministic_policy_class(native_class):
    """Distinct worker subclass; no replacement of native kernels/loss/optimizer.

    Factory itself does not change Torch state. The flags are activated inside
    the isolated Ray policy actor BEFORE its native constructor/model setup.
    Unsupported deterministic operators fail, never warn-only or fall back.
    """
    class DeterministicPRMPolicyWorker(native_class):
        def __init__(self, *args, **kwargs):
            import torch
            _require_training_numerics_environment()
            self._prm_cuda_initialized_before_native_constructor = torch.cuda.is_initialized()
            torch.use_deterministic_algorithms(True, warn_only=False)
            super().__init__(*args, **kwargs)
            _require_training_numerics_state()

        def get_prm_numerics_receipt(self):
            """Read-only actor-local state; no CUDA/device query or collective."""
            import os
            import torch
            _require_training_numerics_state()
            return {
                'numerics': GRADIENT_DEFAULTS['numerics'],
                'environment': {key: os.environ.get(key) for key in TRAINING_NUMERICS_ENV},
                'torch_deterministic': torch.are_deterministic_algorithms_enabled(),
                'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                'cuda_initialized_before_native_constructor': self._prm_cuda_initialized_before_native_constructor,
                'cuda_initialized': torch.cuda.is_initialized(),
                'model_initialized': self.model is not None,
                'rank': self._rank,
                'pid': os.getpid(),
            }

        def init_model(self, *args, **kwargs):
            _require_training_numerics_state()
            result = super().init_model(*args, **kwargs)
            _require_training_numerics_state()
            return result

        def forward(self, *args, **kwargs):
            _require_training_numerics_state()
            return super().forward(*args, **kwargs)

        def forward_backward(self, *args, **kwargs):
            _require_training_numerics_state()
            return super().forward_backward(*args, **kwargs)

    return DeterministicPRMPolicyWorker


def deterministic_policy_actor(native_actor):
    import ray
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, FSDPPolicyWorkerBase
    if native_actor is not PolicyWorker:
        raise ValueError('PRM deterministic worker only qualifies the native FSDP policy actor')
    _require_training_numerics_environment()
    # Explicit Ray runtime_env is applied before worker imports/CUDA startup.
    # Do not rely only on inherited driver environment or alter Ref/Critic.
    return ray.remote(num_gpus=1, runtime_env={'env_vars': dict(TRAINING_NUMERICS_ENV)})(
        deterministic_policy_class(FSDPPolicyWorkerBase))


def gradient_config_for_plan(plan):
    value = plan.get('gradient_probe')
    if value is None:
        return None
    if (plan.get('process_reward') is None or not isinstance(value, dict)
            or value != GRADIENT_DEFAULTS
            or type(value.get('interval')) is not int
            or type(value.get('version')) is not int
            or value.get('first_step') is not True):
        raise ValueError('Exact opt-in PRM gradient probe configuration required')
    return {**value, 'components': list(value['components'])}


def make_config(**overrides):
    value = {**DEFAULTS, **overrides}
    validate(value)
    return value


def validate(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULTS):
        raise ValueError('Exact versioned PRM config required')
    if value['protocol'] != 'v5' or value['window'] != 10 or value['votes'] != 3:
        raise ValueError('v5 uses ten-turn windows and three independent requests')
    for name in ('window', 'votes', 'concurrency', 'max_pending_windows', 'max_attempts'):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError('Invalid PRM integer: ' + name)
    if value['max_attempts'] != 2 or value['concurrency'] > 256 or value['max_pending_windows'] > 1024:
        raise ValueError('Unbounded online PRM request budget')
    for name in ('weight', 'request_timeout_s', 'window_timeout_s', 'finish_timeout_s', 'requests_per_second'):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or value[name] <= 0:
            raise ValueError('Invalid PRM numeric config: ' + name)
    if (value['weight'] > 1 or value['window_timeout_s'] > 600 or value['finish_timeout_s'] > 600
            or value['request_timeout_s'] > value['window_timeout_s']):
        raise ValueError('Unsafe PRM weight/deadline')
    if not isinstance(value['model'], str) or not re.fullmatch(r'gemini-[A-Za-z0-9._-]+', value['model']):
        raise ValueError('Explicit Gemini model required; no fallback')
    return dict(value)


def config_for_plan(plan):
    value = plan.get('process_reward')
    return None if value is None else validate(value)


def check_training_config(plan, cfg):
    value = config_for_plan(plan)
    validate_decay(plan.get('process_decay'))
    if plan.get('process_decay') is not None and (value is None or value['weight'] != 0.5):
        raise ValueError('Decay requires the original 0.5 process configuration')
    gradient = gradient_config_for_plan(plan)
    interval = getattr(cfg.generator, 'ocrl_gradient_probe_interval', 0)
    if (type(interval) is not int or interval != (0 if gradient is None else gradient['interval'])):
        raise ValueError('Gradient probe schedule differs from the immutable plan')
    enabled = getattr(cfg.generator, 'ocrl_packed_prm_enabled', False)
    if type(enabled) is not bool or enabled != (value is not None):
        raise ValueError('Packed PRM opt-in differs from the immutable plan')
    expected = 0. if value is None else value['weight']
    if cfg.generator.ocrl_process_weight != expected:
        raise ValueError('PRM coefficient differs from the immutable plan')
    if value is not None and (plan.get('harness') != 'append-opencode-tools-v1'
            or plan.get('causal_prefix') is not True or cfg.generator.ocrl_causal_prefix is not True):
        raise ValueError('Online v5 is wired only for the append causal training path')
    return value


DECAY_V1 = {'version': 1, 'kind': 'polynomial', 'start_step': 20,
    'duration_steps': 10, 'power': 3, 'initial_weight': 0.5, 'final_weight': 0.0,
    'step_basis': 'consumer_optimizer_global_step'}


def validate_decay(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or value != DECAY_V1
            or any(type(value[k]) is not int for k in ('version','start_step','duration_steps','power'))):
        raise ValueError('Unexpected immutable PRM decay schedule')
    return dict(value)


def decay_weight(value, global_step, *, constant_weight=0.5):
    schedule = validate_decay(value)
    if schedule is None:
        return float(constant_weight)
    if type(global_step) is not int or global_step < 0:
        raise ValueError('PRM decay needs the native optimizer step')
    progress = min(1.0, max(0.0, (global_step - schedule['start_step']) / schedule['duration_steps']))
    return schedule['initial_weight'] * (1.0 - progress) ** schedule['power']
