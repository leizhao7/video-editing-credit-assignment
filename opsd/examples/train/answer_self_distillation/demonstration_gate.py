"""Qualify complete trajectories before native batch admission; never mask turns.

The original rollout, tools, token packing and validation methods stay inherited.
Trusted provider refusals and exhausted malformed provider actions are filtered
as whole trajectories. Provenance/configuration/storage errors remain fatal.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor, CancelledError
from pathlib import Path
import threading
import time

from .answers import require
from .turn_demonstrations import DemonstrationRefused, DemonstrationUnavailable, TurnDemonstrations, contexts_for_row

FORMAT = "whole-trajectory-demo-availability-v2"


class DemonstrationAdmission:
    def __init__(self, plan, tokenizer):
        self.plan = plan
        self.demonstrations = TurnDemonstrations(plan, tokenizer)
        self.pool = ThreadPoolExecutor(max_workers=self.demonstrations.cfg["concurrency"],
                                       thread_name_prefix="opsd-admission-demo")

    async def qualify(self, output):
        from opencode_rl.store import atomic_json
        from opencode_rl.capacity import durable_io
        rows = output["ocrl_rows"]
        require(len(rows) == len(output["prompt_token_ids"]) == len(output["response_ids"]) == 1,
                "Demo admission requires the unchanged n=1 packed output")
        row = rows[0]
        sequence = list(output["prompt_token_ids"][0]) + list(output["response_ids"][0])
        contexts = await asyncio.to_thread(contexts_for_row, self.plan, 0, row, sequence)
        require(contexts, "Cannot admit a trajectory with no original actions")
        started, abort = time.monotonic(), threading.Event()
        jobs = [self.pool.submit(self.demonstrations.one, context, abort) for context in contexts]
        futures = [asyncio.wrap_future(job) for job in jobs]
        proof = None
        try:
            completed = await asyncio.gather(*futures, return_exceptions=True)
        except BaseException:
            abort.set()
            for job in jobs:
                job.cancel()
            await asyncio.gather(*futures, return_exceptions=True)
            raise
        excluded = (DemonstrationRefused, DemonstrationUnavailable)
        # An output refusal cannot hide a concurrent storage/provenance failure.
        # Cancellation of unsent siblings is expected after a trusted exclusion.
        fatal = next((item for item in completed if isinstance(item, BaseException) and not
                      isinstance(item, (*excluded, CancelledError, asyncio.CancelledError))), None)
        if fatal is not None:
            raise fatal
        refused = next((item for item in completed if isinstance(item, excluded)), None)
        if refused is not None:
            proof = {"format": FORMAT, "trajectory_id": row["trajectory_id"], "task": row["task_id"],
                "reason": refused.reason, "request_id": refused.request_id,
                "whole_trajectory_excluded": True, "model_reward_zero": False,
                "original_request_ids": [context.span["request_id"] for context in contexts],
                "elapsed_seconds": time.monotonic() - started}
            if isinstance(refused, DemonstrationUnavailable):
                proof.update(output_attempts=refused.attempts, last_error_type=refused.error_type)
            path = Path(self.plan["run_root"]) / "opsd_demo_exclusions" / (row["trajectory_id"] + ".json")
            await durable_io(atomic_json, path, proof)
        else:
            fatal = next((item for item in completed if isinstance(item, BaseException)), None)
            if fatal is not None:
                raise fatal
            require(len(completed) == len(contexts), "Not every original turn has a demonstration")
        # Trusted coordinator metadata only; student tokens, masks, images and
        # outcome labels are never changed by demonstration qualification.
        row["opsd_demo_admission"] = {"format": FORMAT, "ready": proof is None,
            "request_ids": [context.span["request_id"] for context in contexts], "refusal": proof,
            "qualification_seconds": time.monotonic() - started}
        return output

    async def close(self):
        await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)


async def close_admission(trainer):
    generator = getattr(trainer, "generator", None)
    admission = getattr(generator, "demo_admission", None)
    if admission is not None:
        await admission.close()
        await generator.commit()


def qualified_generator_type(base):
    class QualifiedGenerator(base):
        async def generate(self, generator_input):
            output = await super().generate(generator_input)
            output = await self.demo_admission.qualify(output)
            await self.commit()
            return output
    # Validation does not need demonstrations and remains the original method.
    return QualifiedGenerator
