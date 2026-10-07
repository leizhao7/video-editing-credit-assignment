"""Explicit user waiver of extra prelaunch experiments, never runtime checks.

This is not a qualification report. The original OpenCode launcher/validator
remains untouched; this distinct harness records the user's exact new request.
"""
from pathlib import Path

from opencode_rl.assets import files_hash
from opencode_rl.contract import ParityError, digest


def authorization(contract):
    return {'kind': 'append_user_requested_direct_start', 'instruction': '直接开始run',
        'initial_weights': 'base', 'waived': ['additional_prelaunch_qualification'],
        'contract_sha256': contract.sha256, 'qualification_complete': False,
        'runtime_parity_guards_enabled': True}


def validate_authorization(value, contract):
    if not isinstance(value, dict) or value != authorization(contract):
        raise ParityError('Direct start needs exact, snapshot-bound user authorization; not a readiness claim')


def check_direct_plan(plan, contract, cfg):
    """Preserve every native formal-run topology/config guard after the waiver."""
    import opencode_rl
    validate_authorization(plan.get('direct_start_authorization'), contract)
    package = Path(opencode_rl.__file__).parent
    actual = files_hash(package, [p.name for p in package.iterdir() if p.suffix in ('.py', '.mjs')])
    if actual != contract.integration_sha256 or digest(plan['config']) != contract.config_sha256:
        raise ParityError('Training image/runtime source differs from inference')
    if (plan.get('mode') != 'train' or plan.get('harness') != 'append-opencode-tools-v1'
            or plan.get('causal_prefix') is not True or cfg.generator.ocrl_causal_prefix is not True):
        raise ParityError('Direct start is only the explicit append-only formal training mode')
    t, g = cfg.trainer, cfg.generator
    if (t.train_batch_size != 16 or t.policy_mini_batch_size != 16 or
            g.n_samples_per_prompt != 8 or plan.get('eval_interval') != 10):
        raise ParityError('Formal GRPO requires batch16/K8/val10')
    if g.ocrl_process_weight != 0:
        raise ParityError('Outcome-only run cannot silently enable a missing PRM')
    if t.ckpt_interval != 2 or t.max_ckpts_to_keep != 2:
        raise ParityError('Expected checkpoint every two steps with two rolling copies')
    if not plan.get('validation'):
        raise ParityError('Missing frozen validation plan')
    if (t.eval_before_train or t.eval_interval != 10 or t.placement.colocate_all or
            not t.fully_async.enabled or t.fully_async.simulate_training or
            cfg.data.dataloader.num_workers != 0 or g.inference_engine.weight_sync_backend != 'nccl' or
            t.algorithm.max_seq_len != contract.max_context):
        raise ParityError('Unexpected training topology/eval/sequence configuration')
    if (g.eval_n_samples_per_prompt != 1 or g.eval_sampling_params.temperature != contract.temperature or
            g.eval_sampling_params.top_p != contract.top_p or g.eval_sampling_params.top_k != contract.top_k or
            g.eval_sampling_params.max_generate_length != contract.max_output):
        raise ParityError('Validation sampling differs from the shared rollout contract')
    if (len(plan['engines']) != 4 or len({e['url'] for e in plan['engines']}) != 4 or
            t.placement.policy_num_gpus_per_node != 4 or t.placement.policy_num_nodes != 1):
        raise ParityError('Expected four independent inference engines and four training GPUs')
    if (t.resume_mode != 'none' or t.resume_path or t.max_training_steps != plan['requested_steps'] or
            not 1 <= t.max_training_steps <= 50):
        raise ParityError('User selected a new bounded base run, not an implicit checkpoint resume')
