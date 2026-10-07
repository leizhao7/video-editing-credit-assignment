"""Reuse the deployed v15 extensions without pretending to resume GRPO state."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

from .answers import require

EXTENSIONS = {"preparation": "typed-proof-scored-first-v1", "delivery": "grade-inputs-before-optional-preview-v1",
              "grading_boundary": "trusted-candidate-finish-refusal-v1",
              "checkpoint_publication": "bounded-rolling-checkpoint-publication-v1"}


def verify_frozen_extensions(plan):
    from append_resume_guard import adapter_hash
    requested = plan["opsd"]["runtime_extensions"]
    require(all(requested.get(k) == v for k, v in EXTENSIONS.items()), "Missing proven v15 runtime extensions")
    require(requested["adapter_sha256"] == adapter_hash(), "Frozen runtime adapter differs from the source run")


def generator_class():
    from append_resume_preparation import PreparedAppendGenerator, prepare_deduplicated
    from append_resume_delivery import rollout_task
    from .generator import AnswerOpsdGenerator

    class FrozenAnswerOpsdGenerator(AnswerOpsdGenerator, PreparedAppendGenerator):
        def __init__(self, **kwargs):
            # K=1 constructor only. The original _completed_rollout and every
            # _one/generate/evaluation_one implementation remain inherited.
            AnswerOpsdGenerator.__init__(self, **kwargs)
            self.rollout = rollout_task
            self.prepare = prepare_deduplicated

    require(FrozenAnswerOpsdGenerator._completed_rollout is PreparedAppendGenerator._completed_rollout,
            "Lost the deployed early-grading/preparation owner")
    return FrozenAnswerOpsdGenerator


def persistence_class():
    from append_resume_checkpoint import BackgroundCheckpointPersistence
    from opencode_rl.train_persistence import RunPersistence

    class OpsdBackgroundPersistence(BackgroundCheckpointPersistence):
        def __init__(self, plan, logs, checkpoints):
            verify_frozen_extensions(plan)
            # The old constructor requires plan.resume, which would falsely
            # label this base-start OPSD as a checkpoint resume. Initialize its
            # small scheduler state explicitly; reuse every publication method.
            RunPersistence.__init__(self, plan, logs, checkpoints)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opsd-checkpoint-publication")
            self._pending = None
            self._lock = threading.RLock()
            self._closed = False

    return OpsdBackgroundPersistence
