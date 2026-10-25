"""Distinct SkyRL entrypoint; inherit unchanged optimizer/ckpt/eval behavior."""
import asyncio
import json
from pathlib import Path
import sys

from opencode_rl.contract import ParityError
from opencode_rl.train_entrypoint import check_plan as native_check_plan, grading_payload
from opencode_rl.capacity import durable_io
from .contract import AppendContract, validate_runtime


def check_plan(plan, contract, cfg):
    validate_runtime(plan['config'], contract)
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
            return AppendGenerator(dataset=self.train_dataset, contract=self.contract, config=p['config'],
                app=modal.App.lookup(p['sandbox_app'], create_if_missing=True), image_id=p['sandbox_image'],
                engines=[{**e, 'request_gate': asyncio.Semaphore(capacity['streams_per_engine'])} for e in p['engines']],
                root=p['run_root'], grader=grade, commit=commit, k=cfg.generator.n_samples_per_prompt,
                concurrency=p['rollout_concurrency'], preparation_concurrency=capacity['preparation_concurrency'],
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
