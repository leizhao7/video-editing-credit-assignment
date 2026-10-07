"""Distinct SkyRL entrypoint; inherit unchanged optimizer/ckpt/eval behavior."""
import asyncio
import json
from pathlib import Path
import sys

from opencode_rl.contract import ParityError
from opencode_rl.train_entrypoint import check_plan as native_check_plan, grading_payload
from opencode_rl.capacity import durable_io
from .contract import AppendContract, validate_runtime


def prm_event_loop_receipt():
    """Fail before model initialization if Ray ignored the scoped loop policy."""
    import os
    if os.environ.get('RAY_USE_UVLOOP') != '0':
        raise ParityError('PRM trainer requires RAY_USE_UVLOOP=0 before Ray worker startup')
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    try:
        if not isinstance(loop, asyncio.SelectorEventLoop):
            raise ParityError('PRM trainer event loop is not the qualified stdlib selector loop')
        return {'ray_use_uvloop': '0', 'policy': type(policy).__module__+'.'+type(policy).__name__,
                'loop': type(loop).__module__+'.'+type(loop).__name__, 'pid': os.getpid(),
                'scope': 'Ray worker startup policy before training loop creation'}
    finally:
        loop.close()


def check_plan(plan, contract, cfg):
    validate_runtime(plan['config'], contract)
    if plan.get('gradient_probe') is not None or getattr(cfg.generator, 'ocrl_gradient_probe_interval', 0):
        from prm_v2.live_config_v5 import check_training_config
        check_training_config(plan, cfg)
        if plan.get('direct_start_authorization') is not None:
            raise ParityError('Gradient diagnostics require fresh qualification, not an old direct-start waiver')
    if plan.get('process_reward') is not None:
        from prm_v2.live_config_v5 import check_training_config
        from prm_training_source import verify_runtime
        check_training_config(plan, cfg)
        verify_runtime(plan)
        import os
        if not os.environ.get('GOOGLE_AI_STUDIO_API_KEY'):
            raise ParityError('V5 PRM secret is missing; refusing a silently outcome-only launch')
    direct = plan.get('direct_start_authorization') is not None
    if direct:
        from .direct_start import check_direct_plan
        check_direct_plan(plan, contract, cfg)
    else:
        native_check_plan(plan, contract, cfg)
    if plan.get('harness') != 'append-opencode-tools-v1' or not plan.get('causal_prefix'):
        raise ParityError('Append-only training must explicitly enable one-packet causal training')
    if plan.get('mode') == 'train' and not direct:
        report = json.loads(Path(plan['append_qualification']).read_bytes())
        if (report.get('contract_sha256') != contract.sha256 or
                any(report.get(key) is not True for key in ('real_token_image_parity', 'raw_logprob_parity',
                    'loss_gradient_parity', 'all_rank_collectives', 'sandbox_tools_passed'))):
            raise ParityError('Real append-loop GPU/tool qualification is still required')
        if plan.get('gradient_probe') is not None:
            from opencode_rl.contract import digest
            if (report.get('prm_gradient_probe_passed') is not True or
                    report.get('gradient_probe_config_sha256') != digest(plan['gradient_probe']) or
                    report.get('process_source_sha256') != plan['process_source_sha256']):
                raise ParityError('Current-source all-rank gradient probe qualification is still required')


def classes():
    from opencode_rl.train_entrypoint import classes as native_classes
    from skyrl.train.entrypoints.main_base import BasePPOExp
    from opencode_rl.grading_client import OutcomeGrader, modal_caller
    from opencode_rl.capacity import CoalescedCommit
    from opencode_rl.train_capacity import training_capacity
    from .generator import AppendGenerator
    Config, NativeExperiment = native_classes()

    class AppendExperiment(NativeExperiment):
        def __init__(self, cfg):
            self.plan = json.loads(Path(cfg.generator.ocrl_plan).read_bytes())
            self.contract = AppendContract(**self.plan['contract'])
            check_plan(self.plan, self.contract, cfg)
            if self.plan.get('process_reward') is not None:
                from opencode_rl.store import atomic_json
                atomic_json(Path(self.plan['run_root']) / 'event_loop.json', prm_event_loop_receipt())
            # NativeExperiment's initializer loads a different contract type.
            # Reuse its dataset/trainer/validation methods, not that initializer.
            BasePPOExp.__init__(self, cfg)

        def get_generator(self, cfg, tokenizer, inference_engine_client):
            import modal
            p = self.plan
            volume = modal.Volume.from_name(p['logs_volume'])
            commit = CoalescedCommit(volume.commit.aio)
            function = modal.Function.from_name(p['grader_app'], p['grader_function'])
            capacity = training_capacity()
            if p.get('capacity') != capacity:
                raise ParityError('Launcher and training queue capacity differ')
            outcome = OutcomeGrader(modal_caller(function), grader_hashes=p['grader_hashes'],
                concurrency=capacity['judge_concurrency'], review_policy=p.get('grading_review_policy', 'reviewed'),
                submission_validation=p.get('submission_validation'))
            async def grade(task, row, attempt):
                payload = await durable_io(grading_payload, task, row, attempt)
                return await outcome.grade(payload, attempt / 'grading', row['source_hashes'], commit)
            process_judge = None
            if p.get('process_reward') is not None:
                from prm_v2.live_judge_v5 import OnlineJudge
                from prm_v2.sandbox_admission_v5 import configure
                from opencode_rl.store import atomic_json
                configure(app_name=p['sandbox_app'], owner=p['run_root'], policy=p['sandbox_admission'])
                async def save_process(path, value):
                    await durable_io(atomic_json, path, value)
                process_judge = OnlineJudge(p['process_reward'], save=save_process)
            return AppendGenerator(process_judge=process_judge, dataset=self.train_dataset, contract=self.contract, config=p['config'],
                app=modal.App.lookup(p['sandbox_app'], create_if_missing=True), image_id=p['sandbox_image'],
                engines=[{**e, 'request_gate': asyncio.Semaphore(capacity['streams_per_engine'])} for e in p['engines']],
                root=p['run_root'], grader=grade, commit=commit, k=cfg.generator.n_samples_per_prompt,
                concurrency=p['rollout_concurrency'], preparation_concurrency=capacity['preparation_concurrency'],
                residency_limits=p['rollout_residency']['per_engine_limits'],
                read_cache_bytes=capacity['read_cache_bytes'])
    return Config, AppendExperiment


def main():
    import ray
    from skyrl.train.utils import validate_cfg
    from skyrl.train.utils.utils import initialize_ray
    Config, _ = classes()
    cfg = Config.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    @ray.remote(num_cpus=1, max_retries=0)
    def run(config):
        _, Experiment = classes()
        Experiment(config).run()
    ray.get(run.remote(cfg))


if __name__ == '__main__':
    main()
