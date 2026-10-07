"""Import/source-only preflight in the actual training venv; no GPU execution.

Run with /opt/SkyRL/.venv/bin/python -m prm_v2.cloud_preflight_v5 --plan -.
The CLI hides CUDA before importing Torch/SkyRL and accepts immutable plan JSON
on stdin. It does not initialize a strategy, model, Ray, process group, optimizer,
provider client, or Sandbox. A successful report is NOT hardware qualification.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

VERSION = 'prm-cloud-import-preflight-v1'
TRAINING_PYTHON = '/opt/SkyRL/.venv/bin/python'
RUNTIME_MODULES = ('prm_v2.live_judge_v5', 'prm_v2.live_packet_v5',
                   'prm_v2.gradient_probe_v5', 'prm_v2.gradient_source_guard_v5')


class PreflightError(RuntimeError):
    pass


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _module_receipt(module):
    path = getattr(module, '__file__', None)
    if not isinstance(path, str) or not Path(path).is_file():
        raise PreflightError('Imported module has no inspectable source: ' + str(getattr(module, '__name__', '?')))
    return {'module': module.__name__, 'file': path, 'sha256': _sha_file(path)}


def _idle_boundary():
    """Passive state checks only: never ask CUDA for devices or initialize Ray."""
    torch = sys.modules.get('torch')
    ray = sys.modules.get('ray')
    if torch is not None:
        if torch.cuda.is_initialized():
            raise PreflightError('CUDA was initialized in an import-only preflight process')
        if torch.distributed.is_initialized():
            raise PreflightError('Distributed process group was initialized')
    if ray is not None and ray.is_initialized():
        raise PreflightError('Ray was initialized')
    return {'cuda_initialized': False, 'distributed_initialized': False, 'ray_initialized': False}


def _environment():
    if str(Path(sys.executable).absolute()) != TRAINING_PYTHON or sys.version_info[:2] != (3, 12):
        raise PreflightError('Use the image training Python 3.12 venv, not the Modal host interpreter')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PreflightError('Import preflight requires explicit CUDA_VISIBLE_DEVICES=""')
    return {'python_executable': sys.executable, 'python_version': sys.version,
            'visible_devices': '', **_idle_boundary()}


def _native_bindings(guard):
    names = (guard.PPO, guard.TORCH_UTILS, guard.OFFPOLICY, guard.STRATEGY, guard.WORKER, guard.TRAINER)
    modules = {name: importlib.import_module(name) for name in names}
    strategy_class = modules[guard.STRATEGY].FSDPStrategy
    # No __init__: the carrier only exposes the actual class's bound backward.
    # Never claim that an initialized FSDP worker or its sharding was tested.
    strategy = object.__new__(strategy_class)
    carrier = SimpleNamespace(strategy=strategy, policy_loss_fn=modules[guard.PPO].rollout_is_policy_loss)
    worker_receipt = guard.verify_worker(carrier)
    normalizer = modules[guard.TRAINER].RayPPOTrainer._normalize_advantages
    normalizer_receipt = guard.verify_normalizer(normalizer)
    return {'scope': 'import-only native binding carrier; no initialized worker or FSDP layout',
            'backward_owner': strategy_class.__module__ + '.' + strategy_class.__qualname__,
            'worker_guard': worker_receipt, 'normalizer_guard': normalizer_receipt,
            'modules': [_module_receipt(modules[name]) for name in names]}


def run_preflight(plan, *, root='/opt'):
    """Return structured success/failure without mutating a plan or writing files.

    Exceptions are recorded by check and fail closed. No exception or absent
    check becomes a passed flag. Imported dependencies may print their own
    diagnostics; the CLI redirects those to stderr to keep stdout one JSON.
    """
    started = time.monotonic()
    report = {'version': VERSION, 'status': 'failed', 'checks': [],
              'training_qualified': False, 'gpu_probe_performed': False,
              'prm_gradient_probe_passed': False, 'optimizer_steps': 0,
              'provider_requests': 0, 'sandbox_created': False,
              'preflight_source_sha256': _sha_file(__file__)}

    def check(name, function):
        check_started = time.monotonic()
        row = {'name': name, 'passed': False}
        report['checks'].append(row)
        try:
            value = function()
        except Exception as exc:
            row['error'] = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
            raise PreflightError(name) from exc
        finally:
            row['seconds'] = time.monotonic() - check_started
        row.update(passed=True, evidence=value)
        return value

    try:
        check('training_interpreter_and_idle_boundary', _environment)
        if type(plan) is not dict:
            raise PreflightError('Immutable plan must be a JSON object')
        source = importlib.import_module('prm_training_source')
        verified = check('mounted_manifest_and_plan', lambda: source.verify_runtime(plan, root=root))
        report['process_source_sha256'] = verified['source_sha256']
        imported = {}
        def runtime_imports():
            for name in RUNTIME_MODULES:
                imported[name] = importlib.import_module(name)
            return [_module_receipt(imported[name]) for name in RUNTIME_MODULES]
        check('prm_runtime_imports', runtime_imports)
        def image_fixes():
            results = {}
            for name in ('opencode_rl.install_training_fixes', 'opencode_rl.install_startup_fixes'):
                module = importlib.import_module(name)
                receipt = module.verify_installed()
                results[name] = {'module': _module_receipt(module), 'installation_receipt': receipt}
            return results
        check('installed_image_source_fixes', image_fixes)
        def dependencies():
            values = {}
            for name in ('torch', 'aiohttp', 'PIL.Image'):
                module = importlib.import_module(name)
                values[name] = {**_module_receipt(module), 'version': str(getattr(module, '__version__', 'unknown'))}
            return values
        check('training_dependency_imports', dependencies)
        check('actual_native_source_bindings', lambda: _native_bindings(imported['prm_v2.gradient_source_guard_v5']))
        check('post_import_manifest_and_origins', lambda: source.verify_runtime(plan, root=root))
        check('post_import_idle_boundary', _idle_boundary)
        report['status'] = 'passed'
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
        # Detect import side effects even when a prior source/import check failed.
        try:
            report['final_idle_boundary'] = _idle_boundary()
        except Exception as boundary_error:
            report['final_idle_boundary'] = {'error': type(boundary_error).__name__ + ': ' + str(boundary_error)[:500]}
    report['seconds'] = time.monotonic() - started
    return report


def _strict_plan(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PreflightError('Duplicate plan JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise PreflightError('Nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='-', help='Frozen JSON plan path; - reads stdin')
    args = parser.parse_args(argv)
    # Process-local only, before importing any training dependencies. Do not
    # change other flags or initialize runtime services to make a check pass.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        raw = sys.stdin.read(8_000_001) if args.plan == '-' else Path(args.plan).read_text()
        if len(raw) > 8_000_000:
            raise PreflightError('Preflight plan exceeds bounded JSON input')
        plan = _strict_plan(raw)
        with contextlib.redirect_stdout(sys.stderr):
            report = run_preflight(plan)
    except Exception as exc:
        report = {'version': VERSION, 'status': 'failed', 'checks': [],
                  'training_qualified': False, 'gpu_probe_performed': False,
                  'failure': {'type': type(exc).__name__, 'message': str(exc)[:1000]}}
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
