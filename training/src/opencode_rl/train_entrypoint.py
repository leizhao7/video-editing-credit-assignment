"""Independent SkyRL training entrypoint, explicit smoke vs production gate.

No future annotations: SkyRL builds nested configs from concrete dataclass types.
No legacy generator import/registration or production class monkey-patching.
"""
import asyncio
import json
from pathlib import Path
import sys

from .contract import Contract, ParityError, digest
from .readiness import require_training_ready
from .train_dataset import PreparedDataset
from .capacity import durable_io


def grading_payload(task, row, attempt):
    """Read frozen grading artifacts off-loop; previews may be up to 128MiB."""
    names = {'s_1': 'selection.json', 's_2': 'order.json', 's_3': 'cuts.json', 's_4e': 'edit.json'}
    output = attempt / names[row['stage']]
    preview = attempt / 'preview.mp4'
    return {'task': task, 'stage': row['stage'], 'source_root': row['source_root'],
            'deliverable_json': output.read_text() if output.exists() else '',
            'preview_bytes': preview.read_bytes() if row['stage'] == 's_4e' and preview.exists() else None}


def check_plan(plan, contract, cfg):
    from .assets import files_hash
    package = Path(__file__).parent
    actual = files_hash(package, [p.name for p in package.iterdir() if p.suffix in ('.py', '.mjs')])
    if actual != contract.integration_sha256 or digest(plan['config']) != contract.config_sha256:
        raise ParityError('Training image/runtime source differs from inference')
    if type(plan.get('causal_prefix', False)) is not bool or cfg.generator.ocrl_causal_prefix != plan.get('causal_prefix', False):
        raise ParityError('Causal training mode must match its explicit launch plan')
    if plan.get('mode') == 'smoke':
        if not 1 <= cfg.trainer.max_training_steps <= 2:
            raise ParityError('Explicit smoke is limited to two optimizer steps')
    elif plan.get('mode') == 'train':
        if plan.get('direct_start_authorization') is not None:
            from .readiness import require_direct_start_authorization
            require_direct_start_authorization(plan['direct_start_authorization'], contract)
        else:
            report = json.loads(Path(plan['readiness_report']).read_bytes())
            require_training_ready(report, contract, process_enabled=cfg.generator.ocrl_process_weight != 0)
            if report.get('grader_hashes') != plan['grader_hashes'] or report.get('prepared_sha256') != plan['prepared_sha256']:
                raise ParityError('Training readiness does not cover this grader/prepared release')
        if (cfg.trainer.train_batch_size != 16 or cfg.trainer.policy_mini_batch_size != 16 or
                cfg.generator.n_samples_per_prompt != 8 or plan.get('eval_interval') != 10):
            raise ParityError('Formal GRPO requires batch16/K8/val10')
    else:
        raise ParityError('Training mode must be explicitly smoke or train')
    if plan.get('process_reward') is not None or getattr(cfg.generator, 'ocrl_packed_prm_enabled', False):
        from prm_v2.live_config_v5 import check_training_config
        check_training_config(plan, cfg)
    elif cfg.generator.ocrl_process_weight != 0:
        raise ParityError('This outcome integration does not silently substitute missing online PRM')
    if cfg.trainer.ckpt_interval != 2 or cfg.trainer.max_ckpts_to_keep != 2:
        raise ParityError('Expected checkpoint every two steps with two rolling copies')
    interval = plan.get('eval_interval', -1)
    if interval not in (-1, 10) or (interval == 10 and not plan.get('validation')):
        raise ParityError('Missing frozen validation plan')
    if (cfg.trainer.eval_before_train or cfg.trainer.eval_interval != interval or
            cfg.trainer.placement.colocate_all or not cfg.trainer.fully_async.enabled or
            cfg.trainer.fully_async.simulate_training or cfg.data.dataloader.num_workers != 0 or
            cfg.generator.inference_engine.weight_sync_backend != 'nccl' or
            cfg.trainer.algorithm.max_seq_len != contract.max_context):
        raise ParityError('Unexpected training topology/eval/sequence configuration')
    if interval > 0 and (cfg.generator.eval_n_samples_per_prompt != 1 or
            cfg.generator.eval_sampling_params.temperature != contract.temperature or
            cfg.generator.eval_sampling_params.top_p != contract.top_p or
            cfg.generator.eval_sampling_params.top_k != contract.top_k or
            cfg.generator.eval_sampling_params.max_generate_length != contract.max_output):
        raise ParityError('Validation sampling differs from the shared rollout contract')
    if (len(plan['engines']) != 4 or len({e['url'] for e in plan['engines']}) != 4 or
            cfg.trainer.placement.policy_num_gpus_per_node != 4 or
            cfg.trainer.placement.policy_num_nodes != 1):
        raise ParityError('Expected four independent inference engines and four training GPUs')


def classes():
    from .install_startup_fixes import verify_installed
    verify_installed()
    from .train_config import config_class
    from skyrl.train.entrypoints.main_base import BasePPOExp
    from .skyrl_adapter import trainer_classes
    from .train_generator import OpenCodeGenerator
    from .grading_client import OutcomeGrader, modal_caller
    from .capacity import CoalescedCommit
    from .train_capacity import training_capacity
    from .train_persistence import PersistedAsyncTrainerMixin, run_persistence
    from .train_validation import ValidationTrainerMixin
    _, AsyncTrainer = trainer_classes()

    class DurableAsyncTrainer(ValidationTrainerMixin, PersistedAsyncTrainerMixin, AsyncTrainer):
        pass

    OpenCodeConfig = config_class()

    class OpenCodeExperiment(BasePPOExp):
        def __init__(self, cfg):
            self.plan = json.loads(Path(cfg.generator.ocrl_plan).read_bytes())
            self.contract = Contract(**self.plan['contract'])
            check_plan(self.plan, self.contract, cfg)
            super().__init__(cfg)

        def get_train_dataset(self):
            p = self.plan
            result = PreparedDataset(p['prepared_manifest'], p['prepared_bundles'], p['release_path'])
            if result.sha256 != p['prepared_sha256']:
                raise ParityError('Prepared training release changed')
            return result

        def get_eval_dataset(self):
            v = self.plan.get('validation')
            if not v:
                return None
            result = PreparedDataset(v['manifest'], v['bundles'], self.plan['release_path'], split='val')
            if len(result) != 81 or result.sha256 != v['sha256']:
                raise ParityError('Prepared validation release changed')
            return result

        def get_inference_client(self):
            from .control_client import control_client
            urls = [row['url'] for row in self.plan['engines']]
            return control_client(proxy_url=urls[0], server_urls=urls,
                data_parallel_size=1, model_name='qwen35-9b-policy', tokenizer=self.tokenizer)

        def get_generator(self, cfg, tokenizer, inference_engine_client):
            import modal
            from .training import build_generator_output, build_packed_generator_output
            p = self.plan
            volume = modal.Volume.from_name(p['logs_volume'])
            commit = CoalescedCommit(volume.commit.aio)
            function = modal.Function.from_name(p['grader_app'], p['grader_function'])
            capacity = training_capacity()
            if p.get('capacity') != capacity:
                raise ParityError('Launcher and training queue capacity differ')
            outcome = OutcomeGrader(modal_caller(function), grader_hashes=p['grader_hashes'],
                                   concurrency=capacity['judge_concurrency'],
                                   review_policy=p.get('grading_review_policy', 'reviewed'),
                                   submission_validation=p.get('submission_validation'))
            async def grade(task, row, attempt):
                payload = await durable_io(grading_payload, task, row, attempt)
                return await outcome.grade(payload, attempt / 'grading', row['source_hashes'], commit)

            return OpenCodeGenerator(dataset=self.train_dataset, contract=self.contract, config=p['config'],
                app=modal.App.lookup(p['sandbox_app'], create_if_missing=True), image_id=p['sandbox_image'],
                engines=[{**e, 'request_gate': asyncio.Semaphore(capacity['streams_per_engine'])} for e in p['engines']],
                root=p['run_root'], grader=grade, commit=commit, k=cfg.generator.n_samples_per_prompt,
                concurrency=p['rollout_concurrency'],
                preparation_concurrency=capacity['preparation_concurrency'],
                read_cache_bytes=capacity['read_cache_bytes'],
                build_output=build_packed_generator_output if cfg.generator.ocrl_causal_prefix else build_generator_output)

        def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset,
                        inference_engine_client, generator, colocate_pg):
            return DurableAsyncTrainer(cfg=cfg, tracker=tracker, tokenizer=tokenizer,
                train_dataset=train_dataset, eval_dataset=eval_dataset,
                inference_engine_client=inference_engine_client, generator=generator,
                colocate_pg=colocate_pg, persistence=run_persistence(self.plan))

    return OpenCodeConfig, OpenCodeExperiment


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
