"""K=1 constructor adapter; all rollout/evaluation/packing methods are reused.

Do not add private answers here. This object is serialized to rollout workers.
The original GRPO generator keeps its K>=2 guard and remains unchanged.
"""
import asyncio
from functools import partial
from pathlib import Path

from append_agent_rl.contract import validate_runtime
from append_agent_rl.generator import build_append_output
from append_agent_rl.task_rollout import rollout_task
from opencode_rl.contract import ParityError
from opencode_rl.train_generator import OpenCodeGenerator
from opencode_rl.training import prepare_trajectory


class AnswerOpsdGenerator(OpenCodeGenerator):
    def __init__(self, *, dataset, contract, config, app, image_id, engines, root,
                 grader, commit, k=1, concurrency=32, rollout_retries=2,
                 preparation_concurrency=8, read_cache_bytes=64 * 1024 * 1024):
        # Explicit constructor, not a fake K=2 initialization followed by mutation.
        # Runtime method identity is tested against the existing generator.
        validate_runtime(config, contract)
        if (k != 1 or not engines or type(concurrency) is not int or concurrency < 1 or
                type(preparation_concurrency) is not int or preparation_concurrency < 1 or rollout_retries < 1):
            raise ParityError("Answer OPSD requires one trajectory per task and positive bounded concurrency")
        self.dataset, self.contract, self.config = dataset, contract, config
        self.app, self.image_id, self.engines = app, image_id, engines
        self.root, self.grader, self.commit = Path(root), grader, commit
        self.k, self.retries = k, rollout_retries
        self.gate, self.next_engine = asyncio.Semaphore(concurrency), 0
        self.rollout, self.concatenate, self.build_output = rollout_task, None, build_append_output
        self.preparation_gate = asyncio.Semaphore(preparation_concurrency)
        self.read_cache_bytes = read_cache_bytes
        self.prepare = partial(prepare_trajectory, deduplicate_tensors=True)
        self.preparation_policy = "scored_first"
