"""Explicit resume entrypoint around the unchanged v5 rollout/trainer classes."""
from dataclasses import asdict
from pathlib import Path
import sys

from append_resume_guard import (digest, inspect_checkpoint, read_json,
    require, verify_config, verify_plan)


def binding(cfg):
    from append_agent_rl.contract import AppendContract, validate_runtime
    from opencode_rl.assets import files_hash
    import opencode_rl
    plan = read_json(cfg.generator.ocrl_plan)
    root = Path(plan['run_root'])
    source = read_json(root/'resume_source_plan.json')
    audit = read_json(root/'resume_audit.json')
    require(source == read_json(Path(source['run_root'])/'plan.json'), 'Source plan changed')
    verify_plan(plan, source, audit)
    contract = AppendContract(**plan['contract'])
    validate_runtime(plan['config'], contract)
    package = Path(opencode_rl.__file__).parent
    require(files_hash(package, [p.name for p in package.iterdir() if p.suffix in ('.py', '.mjs')]) ==
            contract.integration_sha256, 'Frozen training integration changed')
    checked, trainer, async_state, data_state = inspect_checkpoint(source, plan['resume']['checkpoint'],
                                                                   expected_audit=audit)
    verify_config(trainer['config'], asdict(cfg), plan)
    return plan, contract, checked, trainer, async_state, data_state


class StrictCheckpointLoadMixin:
    """No upstream warn-and-reset fallback for missing/bad dataloader state."""
    def __init__(self, *args, resume_plan, **kwargs):
        self._resume_plan = resume_plan
        super().__init__(*args, **kwargs)

    def load_checkpoints(self):
        plan, contract, audit, trainer, state, data_state = binding(self.cfg)
        require(plan == self._resume_plan, 'Trainer resume binding changed')
        require(not self.has_critic, 'This frozen outcome-GRPO run has no critic')
        # Propagate failures here. Native loading catches Exception and silently
        # resets the dataloader; the fully-async UID/epoch state must survive too.
        self.train_dataloader.load_state_dict(data_state)
        path = plan['resume']['checkpoint']
        self.dispatch.load_checkpoint('policy', str(Path(path)/'policy'),
            load_optimizer_states=True, load_lr_scheduler_states=True)
        from opencode_rl.store import atomic_json
        atomic_json(Path(plan['run_root'])/'resume_loaded.json', {
            'step': audit['step'], 'source_call_id': plan['resume']['source_call_id'],
            'checkpoint': path, 'contract_sha256': contract.sha256, 'audit_sha256': digest(audit),
            'sampler': audit['sampler'], 'optimizer_loaded': True, 'lr_scheduler_loaded': True,
            'inference_weight_sync_complete': False})
        self._ocrl_persistence.logs.commit()
        print(f'Explicit checkpoint resume loaded step={audit["step"]}, epoch={state["epoch"]}, '
              f'consumed={len(state["consumed_uids"])}, filtered={len(state["filtered_uids"])}; '
              'native four-engine weight synchronization follows', flush=True)
        return audit['step'], path, set(state['consumed_uids']), set(state['filtered_uids']), state['epoch']


def classes():
    from append_agent_rl.train_entrypoint import classes as append_classes
    from skyrl.train.entrypoints.main_base import BasePPOExp
    from opencode_rl.skyrl_adapter import trainer_classes
    from opencode_rl.train_persistence import PersistedAsyncTrainerMixin
    from opencode_rl.train_validation import ValidationTrainerMixin
    from append_resume_checkpoint import CheckpointPublicationMixin, run_background_persistence
    Config, AppendExperiment = append_classes()
    _, AsyncTrainer = trainer_classes()

    class ResumeTrainer(StrictCheckpointLoadMixin, ValidationTrainerMixin, CheckpointPublicationMixin,
                        PersistedAsyncTrainerMixin, AsyncTrainer):
        async def train(self):
            import asyncio
            from opencode_rl.store import atomic_json
            loop = asyncio.get_running_loop()
            require(isinstance(loop, asyncio.SelectorEventLoop),
                    'Recovery coordinator must not use the affected uvloop transport')
            atomic_json(Path(self._resume_plan['run_root'])/'transport_runtime.json', {
                'policy': 'stdlib-selector-coordinator-v1',
                'loop_module': type(loop).__module__, 'loop_class': type(loop).__name__,
                'scope': 'training-coordinator-only', 'inference_workers_changed': False})
            self._ocrl_persistence.logs.commit()
            return await super().train()

    class ResumeExperiment(AppendExperiment):
        def _setup_trainer(self):
            import asyncio
            trainer = super()._setup_trainer()
            # Frozen BasePPOExp.run calls asyncio.run immediately after this
            # returns. Set policy AFTER all setup imports (some install uvloop).
            # No global asyncio.run monkeypatch; GPU worker event loops stay unchanged.
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            return trainer

        def __init__(self, cfg):
            self.plan, self.contract, *_ = binding(cfg)
            # Direct-start initializer explicitly forbids checkpoint resumes.
            # Use this distinct, fully validated initializer; do not normalize
            # resume_mode to 'none' or monkeypatch its base-only validator.
            BasePPOExp.__init__(self, cfg)

        def get_generator(self, cfg, tokenizer, inference_engine_client):
            from append_resume_preparation import make_generator
            return make_generator(self, cfg)

        def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset,
                        inference_engine_client, generator, colocate_pg):
            return ResumeTrainer(cfg=cfg, tracker=tracker, tokenizer=tokenizer,
                train_dataset=train_dataset, eval_dataset=eval_dataset,
                inference_engine_client=inference_engine_client, generator=generator,
                colocate_pg=colocate_pg, persistence=run_background_persistence(self.plan), resume_plan=self.plan)

    return Config, ResumeExperiment


def main():
    import ray
    from skyrl.train.utils import validate_cfg
    from skyrl.train.utils.utils import initialize_ray
    Config, _ = classes()
    cfg = Config.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    binding(cfg)  # Fail before initializing the GPU worker groups.
    initialize_ray(cfg)

    @ray.remote(num_cpus=1, max_retries=0)
    def run(config):
        _, Experiment = classes()
        Experiment(config).run()

    ray.get(run.remote(cfg))


if __name__ == '__main__':
    main()
