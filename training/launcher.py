"""Create source-bound plans and commands for fresh, operator-owned runs."""
from pathlib import Path
import hashlib
import json
import re
import tomllib

HERE = Path(__file__).resolve().parent
PYTHON = '/opt/SkyRL/.venv/bin/python'


def bounded_run(name, mode, steps):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Run name must be a simple 1-80 character component')
    if mode not in ('smoke', 'train') or type(steps) is not int or not 1 <= steps <= 50:
        raise ValueError('Select smoke/train and 1-50 optimizer steps')
    if mode == 'smoke' and steps > 2:
        raise ValueError('Smoke runs are capped at two optimizer steps')


def create_plan(config, *, name, image_id, mode, steps, readiness='', direction='forward'):
    bounded_run(name, mode, steps)
    if direction not in ('forward', 'reverse'):
        raise ValueError('Unknown OPSD KL direction')
    from append_agent_rl.assets import from_assets
    from append_agent_rl.contract import configuration, validate_runtime
    from opencode_rl.contract import MODEL_REVISION, digest
    from opencode_rl.train_dataset import PreparedDataset
    from opencode_rl.train_capacity import training_capacity
    from prm_training_source import mounted_manifest, verify_runtime, SHELL
    release = Path('/release')
    receipt = json.loads((release / 'preparation_receipt.json').read_bytes())
    if receipt.get('public_inputs_prepared') is not True:
        raise ValueError('Release has not completed public input preparation')
    identity = receipt['release_sha256']
    datasets = {}
    for split in ('train', 'val'):
        entry = receipt['prepared'].get(split)
        if not entry:
            continue
        folder = release / 'prepared' / split
        dataset = PreparedDataset(folder / 'manifest.json', folder / 'tasks', release / 'release.json',
                                  split=split, expected_release_sha256=identity)
        if dataset.sha256 != entry['sha256'] or len(dataset) != entry['task_count']:
            raise ValueError('Prepared release differs from its receipt')
        datasets[split] = dataset
    if 'train' not in datasets:
        raise ValueError('No prepared training split')
    profile = config['profile']
    if (mode == 'train' or profile.startswith('opsd_')) and 'val' not in datasets:
        raise ValueError('Formal and OPSD runs require a separate film-held-out validation split')
    model = Path('/models') / MODEL_REVISION
    index = json.loads((model / 'model.safetensors.index.json').read_bytes())
    if any(not (model / shard).is_file() for shard in set(index['weight_map'].values())):
        raise ValueError('Pinned model download is incomplete')
    runtime = configuration(context=131072, output=16384, temperature=1., top_p=1., thinking=True, music=True)
    runtime['shell'] = SHELL
    tools = HERE / 'environment/support'
    tools_hash = digest({name: hashlib.sha256((tools / source).read_bytes()).hexdigest()
                         for name, source in (('worker.py', 'observation_worker.py'), ('schema.py', 'tool_schema.py'))})
    contract = from_assets(model, '/opt/opencode_rl', config=runtime,
        sandbox_sha256=digest({'modal_image_id': image_id}), tools_sha256=tools_hash,
        thinking=True, temperature=1., top_p=1., top_k=-1, max_context=131072, max_output=16384)
    validate_runtime(runtime, contract)
    root = Path('/logs/runs') / name
    checkpoints = Path('/checkpoints') / name
    if root.exists() or checkpoints.exists():
        raise ValueError('Run/checkpoint directory exists; use a new name, including for an explicit resume')
    manifest = mounted_manifest()
    capacity = training_capacity()
    plan = {'harness': 'append-opencode-tools-v1', 'mode': mode, 'requested_steps': steps,
        'run_root': str(root), 'checkpoint_root': str(checkpoints), 'model_path': str(model),
        'causal_prefix': True, 'release_path': str(release / 'release.json'), 'release_sha256': identity,
        'prepared_manifest': str(release / 'prepared/train/manifest.json'),
        'prepared_bundles': str(release / 'prepared/train/tasks'), 'prepared_sha256': datasets['train'].sha256,
        'eval_interval': 10 if 'val' in datasets else -1,
        'readiness_report': readiness, 'append_qualification': readiness,
        'logs_volume': config['logs_volume'], 'checkpoint_volume': config['checkpoint_volume'],
        'sandbox_image': image_id, 'sandbox_app': config['app'] + '-rollouts-' + name,
        'config': runtime, 'contract': contract.to_dict(),
        'grader_app': config['app'], 'grader_function': 'grade',
        'grading_review_policy': 'judge_only', 'submission_validation': 'enabled',
        'capacity': capacity, 'rollout_concurrency': capacity['rollout_concurrency'],
        'rollout_residency': {'per_engine_limits': config['residency_limits']},
        'process_source_manifest': manifest, 'process_source_sha256': digest(manifest),
        'public_profile': profile,
        'engines': [{'url': f'http://127.0.0.1:{18100 + i}', 'trace_root': str(root / f'engine_{i}/trace')}
                    for i in range(4)]}
    if 'val' in datasets:
        val = datasets['val']
        plan['validation'] = {'manifest': str(release / 'prepared/val/manifest.json'),
            'bundles': str(release / 'prepared/val/tasks'), 'sha256': val.sha256, 'task_count': len(val)}
    if profile.startswith('prm_'):
        from prm_v2.live_config_v5 import make_config, GRADIENT_DEFAULTS
        from prm_v2.sandbox_admission_v5 import DEFAULT_POLICY
        from copy import deepcopy
        plan.update(process_reward=make_config(), gradient_probe=deepcopy(GRADIENT_DEFAULTS),
                    sandbox_admission=deepcopy(DEFAULT_POLICY))
    if profile.startswith('opsd_'):
        from examples.train.answer_self_distillation.launch import new_plan
        # The recipe creates a new identity; no historical run is impersonated.
        template = {**plan, 'run_root': '/logs/unlaunched-template',
                    'checkpoint_root': '/checkpoints/unlaunched-template',
                    'engines': [{**e, 'trace_root': e['trace_root'].replace(str(root), '/logs/unlaunched-template')}
                                for e in plan['engines']]}
        options = {}
        if profile == 'opsd_suggestions':
            options['privileged_information'] = 'gemini_turn_action'
        else:
            catalog = release / 'private/reference_answers.json'
            options.update(private_answers_path=str(catalog),
                           private_answers_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest())
            if options['private_answers_sha256'] != receipt['prepared'].get('reference_answers_sha256'):
                raise ValueError('Teacher-only answers changed after preparation')
        plan = new_plan(template, run_root=str(root), checkpoint_root=str(checkpoints), mode=mode, steps=steps,
                        runtime_profile='source', direction=direction, **options)
    verify_runtime(plan)
    return plan, contract


def base_command(plan, *, logger='console'):
    values = tomllib.loads((HERE / 'environment/base-overrides.toml').read_text())['overrides']
    values.update({'trainer.policy.model.path': plan['model_path'], 'trainer.ref.model.path': plan['model_path'],
        'trainer.logger': logger, 'trainer.project_name': 'video-editing-credit-assignment',
        'trainer.run_name': Path(plan['run_root']).name})
    if plan['mode'] == 'smoke' and 'opsd' not in plan:
        values.update({'trainer.train_batch_size': '2', 'trainer.policy_mini_batch_size': '2',
            'generator.n_samples_per_prompt': '2', 'trainer.micro_train_batch_size_per_gpu': '1',
            'trainer.micro_forward_batch_size_per_gpu': '1',
            'trainer.fully_async.num_parallel_generation_workers': '4',
            'environment.skyrl_gym.max_env_workers': '4'})
    return [PYTHON, '-m', 'append_agent_rl.train_entrypoint', *[k + '=' + v for k, v in values.items()]]


def command(plan, *, logger='console', resume=''):
    base = base_command(plan, logger=logger)
    path = Path(plan['run_root']) / 'plan.json'
    if 'opsd' in plan:
        from examples.train.answer_self_distillation.launch import training_command
        return training_command(base, path, plan, resume=resume)
    from append_agent_rl.launch import training_command
    return training_command(base, path, plan, steps=plan['requested_steps'], resume=resume)


def require_qualification(plan, contract):
    """Formal training never invents source, GPU, gradient or resume evidence."""
    from opencode_rl.contract import digest
    from opencode_rl.readiness import require_training_ready
    report = json.loads(Path(plan['readiness_report']).read_bytes())
    require_training_ready(report, contract, process_enabled='process_reward' in plan,
                           release_sha256=plan['release_sha256'])
    required = ('real_token_image_parity', 'raw_logprob_parity', 'loss_gradient_parity',
                'all_rank_collectives', 'sandbox_tools_passed')
    if (any(report.get(k) is not True for k in required) or report.get('grader_hashes') != plan['grader_hashes'] or
            report.get('prepared_sha256') != plan['prepared_sha256']):
        raise ValueError('Readiness does not cover the current runtime, grader and prepared release')
    if 'gradient_probe' in plan and (report.get('prm_gradient_probe_passed') is not True or
            report.get('gradient_probe_config_sha256') != digest(plan['gradient_probe']) or
            report.get('process_source_sha256') != plan['process_source_sha256'] or
            report.get('process_config_sha256') != digest(plan['process_reward'])):
        raise ValueError('Current all-rank PRM gradient qualification is required')
