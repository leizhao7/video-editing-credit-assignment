"""Distinct entrypoint; opt-in only, never imported by the old GRPO launcher."""
import asyncio
import json
from pathlib import Path
import sys


def register_objective():
    from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry, PolicyLossRegistry
    from .answers import require
    from .objective import answer_opsd_passthrough
    # classes() executes on driver and Ray experiment actors, whose registry may
    # already have been synchronized. Do not register the same name twice.
    if "answer_opsd" in AdvantageEstimatorRegistry.list_available():
        existing = AdvantageEstimatorRegistry.get("answer_opsd")
        require((existing.__module__, existing.__name__) ==
                (answer_opsd_passthrough.__module__, answer_opsd_passthrough.__name__), "Conflicting OPSD estimator")
    else:
        AdvantageEstimatorRegistry.register("answer_opsd", answer_opsd_passthrough)
    from .topk import LOSS_NAME, selected_token_fallback_forbidden
    if LOSS_NAME in PolicyLossRegistry.list_available():
        existing = PolicyLossRegistry.get(LOSS_NAME)
        require((existing.__module__, existing.__name__) ==
                (selected_token_fallback_forbidden.__module__, selected_token_fallback_forbidden.__name__),
                "Conflicting top-K policy loss")
    else:
        PolicyLossRegistry.register(LOSS_NAME, selected_token_fallback_forbidden)


def classes():
    from .install import verify_extension
    verify_extension()
    from opencode_rl.train_entrypoint import classes as native_classes, grading_payload
    from opencode_rl.capacity import CoalescedCommit, durable_io
    from opencode_rl.grading_client import OutcomeGrader, modal_caller
    from opencode_rl.skyrl_adapter import trainer_classes
    from opencode_rl.train_capacity import training_capacity
    from opencode_rl.train_persistence import PersistedAsyncTrainerMixin, RunPersistence
    from opencode_rl.train_validation import ValidationTrainerMixin
    from append_agent_rl.contract import AppendContract
    from skyrl.train.entrypoints.main_base import BasePPOExp
    from .answers import PrivateAnswers, require
    from .configuration import check_plan
    from .generator import AnswerOpsdGenerator
    from .launch import objective_identity, recipe_source_sha256
    from .trainer import AnswerOpsdTrainerMixin

    register_objective()

    Config, NativeExperiment = native_classes()
    _, ExactAsync = trainer_classes()

    class OpsdReceiptMixin:
        def save(self, trainer, path):
            from opencode_rl.store import atomic_json
            if hasattr(self, "wait_for_publication"):
                self.wait_for_publication()
            # Write BEFORE the existing permanent-copy/checkpoint publication.
            atomic_json(Path(path) / "opsd_resume.json", {"step": int(trainer.global_step),
                "identity": objective_identity(self.plan)})
            return super().save(trainer, path)

    class OpsdPersistence(OpsdReceiptMixin, RunPersistence):
        pass

    class OpsdTrainer(AnswerOpsdTrainerMixin, ValidationTrainerMixin, PersistedAsyncTrainerMixin, ExactAsync):
        async def train(self):
            require(isinstance(asyncio.get_running_loop(), asyncio.SelectorEventLoop),
                    "Training coordinator must retain the v15 selector transport fix")
            return await super().train()

    class OpsdExperiment(NativeExperiment):
        def _setup_trainer(self):
            trainer = super()._setup_trainer()
            # Some setup imports install uvloop. Apply AFTER setup, immediately
            # before BasePPOExp.run's asyncio.run, as in the working v15 fix.
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            return trainer

        def __init__(self, cfg):
            self.plan = json.loads(Path(cfg.generator.ocrl_plan).read_bytes())
            self.contract = AppendContract(**self.plan["contract"])
            check_plan(self.plan, self.contract, cfg)
            require(self.plan["opsd"]["recipe_sha256"] == recipe_source_sha256(), "OPSD recipe changed after plan binding")
            if self.plan["opsd"]["runtime_profile"] == "frozen_v15":
                from .frozen_runtime import verify_frozen_extensions
                verify_frozen_extensions(self.plan)
            BasePPOExp.__init__(self, cfg)
            p = self.plan
            self.private_answers = PrivateAnswers(p["opsd"]["private_answers_path"], p["opsd"]["private_answers_sha256"],
                                                   p["release_path"], self.train_dataset.by_task)

        def get_generator(self, cfg, tokenizer, inference_engine_client):
            import modal
            p = self.plan
            volume = modal.Volume.from_name(p["logs_volume"])
            commit = CoalescedCommit(volume.commit.aio)
            function = modal.Function.from_name(p["grader_app"], p["grader_function"])
            capacity = training_capacity()
            require(p["capacity"] == capacity, "Launcher/worker capacity differs")
            grader_type, generator_type = OutcomeGrader, AnswerOpsdGenerator
            if p["opsd"]["runtime_profile"] == "frozen_v15":
                from .frozen_runtime import generator_class
                from append_refusal_boundary import qualified_grader_type
                grader_type, generator_type = qualified_grader_type(), generator_class()
            outcome = grader_type(modal_caller(function), grader_hashes=p["grader_hashes"],
                concurrency=capacity["judge_concurrency"], review_policy=p.get("grading_review_policy", "reviewed"),
                submission_validation=p.get("submission_validation"))

            async def grade(task, row, attempt):
                payload = await durable_io(grading_payload, task, row, attempt)
                return await outcome.grade(payload, attempt / "grading", row["source_hashes"], commit)

            # No closure/self reference to private_answers is sent to generator actors.
            # Both training and validation call the exact same append rollout method.
            return generator_type(dataset=self.train_dataset, contract=self.contract, config=p["config"],
                app=modal.App.lookup(p["sandbox_app"], create_if_missing=True), image_id=p["sandbox_image"],
                engines=[{**e, "request_gate": asyncio.Semaphore(capacity["streams_per_engine"])} for e in p["engines"]],
                root=p["run_root"], grader=grade, commit=commit, k=1, concurrency=p["rollout_concurrency"],
                preparation_concurrency=capacity["preparation_concurrency"], read_cache_bytes=capacity["read_cache_bytes"])

        def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset,
                        inference_engine_client, generator, colocate_pg):
            import modal
            p = self.plan
            persistence_type, trainer_type = OpsdPersistence, OpsdTrainer
            if p["opsd"]["runtime_profile"] == "frozen_v15":
                from .frozen_runtime import persistence_class
                from append_resume_checkpoint import CheckpointPublicationMixin
                background = persistence_class()
                class FrozenPersistence(OpsdReceiptMixin, background):
                    pass
                class FrozenTrainer(AnswerOpsdTrainerMixin, ValidationTrainerMixin,
                                    CheckpointPublicationMixin, PersistedAsyncTrainerMixin, ExactAsync):
                    async def train(self):
                        require(isinstance(asyncio.get_running_loop(), asyncio.SelectorEventLoop),
                                "Frozen coordinator transport changed")
                        return await super().train()
                persistence_type, trainer_type = FrozenPersistence, FrozenTrainer
            persistence = persistence_type(p, modal.Volume.from_name(p["logs_volume"]),
                                          modal.Volume.from_name(p["checkpoint_volume"]))
            return trainer_type(cfg=cfg, tracker=tracker, tokenizer=tokenizer, train_dataset=train_dataset,
                eval_dataset=eval_dataset, inference_engine_client=inference_engine_client, generator=generator,
                colocate_pg=colocate_pg, persistence=persistence, opsd_plan=p, private_answers=self.private_answers)

    return Config, OpsdExperiment


def main():
    import ray
    from skyrl.train.utils import validate_cfg
    from skyrl.train.utils.utils import initialize_ray
    # The proven Linux selector event loop handles many inference/trace FDs;
    # never inherit an unrelated GUI/platform event-loop policy into Ray.
    if sys.platform == "linux":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    Config, _ = classes()
    cfg = Config.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)

    @ray.remote(num_cpus=1, max_retries=0)
    def run(config):
        _, Experiment = classes()
        Experiment(config).run()

    ray.get(run.remote(cfg))


if __name__ == "__main__":
    main()
