"""Import and source/config preflight in the recovered training interpreter.

No model/optimizer/Ray initialization, provider request or GPU qualification.
"""
import json
import os
from pathlib import Path
import sys


def run(plan):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or sys.executable != '/opt/SkyRL/.venv/bin/python':
        raise ValueError('Use the training interpreter with CUDA_VISIBLE_DEVICES=""')
    if 'process_reward' in plan:
        from prm_v2.cloud_preflight_v5 import run_preflight
        return run_preflight(plan)
    import torch
    import ray
    from prm_training_source import verify_runtime
    from opencode_rl.install_startup_fixes import verify_installed
    from append_agent_rl.contract import AppendContract
    from launcher import command
    verify_installed()
    source = verify_runtime(plan)
    if 'opsd' in plan:
        from examples.train.answer_self_distillation.main_answer_opsd import classes
        from examples.train.answer_self_distillation.configuration import check_plan
    else:
        from append_agent_rl.train_entrypoint import classes, check_plan
    Config, _ = classes()
    cfg = Config.from_cli_overrides(command(plan)[3:])
    check_plan(plan, AppendContract(**plan['contract']), cfg)
    if torch.cuda.is_initialized() or torch.distributed.is_initialized() or ray.is_initialized():
        raise RuntimeError('Import preflight unexpectedly initialized a compute runtime')
    return {'status': 'passed', 'profile': plan['public_profile'], 'source': source,
            'training_qualified': False, 'gpu_probe_performed': False, 'optimizer_steps': 0,
            'provider_requests': 0, 'sandbox_created': False}


if __name__ == '__main__':
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        result = run(json.load(sys.stdin))
    print(json.dumps(result))
    if result.get('status') != 'passed':
        raise SystemExit(1)
