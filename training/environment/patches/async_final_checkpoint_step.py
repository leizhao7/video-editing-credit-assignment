"""Save the fully-async final checkpoint at the last completed step.

``FullyAsyncRayPPOTrainer.global_step`` denotes the step currently being worked
on.  The loop increments it after every completed update so, when the loop
ends, it points at the *next* untrained step.  Upstream's final safety save used
that value directly: completing step 50 produced ``global_step_51`` and a
resume then skipped a step/data cursor.

Rewind the counter once after all loops have stopped and before the final
checkpoint/HF save.  Periodic checkpoints inside the loop are unaffected.
"""

from __future__ import annotations

import py_compile
import sys


TARGET = "/opt/SkyRL/skyrl/train/fully_async_trainer.py"
MARKER = "S12_ASYNC_FINAL_CHECKPOINT_STEP"

OLD_RANGE = '''                for _step_idx in range(self.global_step, (1 + epoch) * self.num_steps_per_epoch + 1):
'''

NEW_RANGE = '''                # S12_ASYNC_MAX_STEP_RANGE: cap the loop itself, not only
                # after an optimizer step.  Resuming a checkpoint already at
                # max_training_steps must not execute one extra update.
                epoch_last_step = (1 + epoch) * self.num_steps_per_epoch
                if self.cfg.trainer.max_training_steps is not None:
                    epoch_last_step = min(
                        epoch_last_step, int(self.cfg.trainer.max_training_steps)
                    )
                for _step_idx in range(self.global_step, epoch_last_step + 1):
'''

OLD = '''        # safety net: always save final checkpoint at end of training.
        if self.cfg.trainer.ckpt_interval > 0:
'''

NEW = '''        # ``global_step`` is the next step to work on here, while checkpoint
        # names/state represent completed optimizer steps.  Keep the final safety
        # save aligned with periodic saves and resume bookkeeping.
        # S12_ASYNC_FINAL_CHECKPOINT_STEP
        self.global_step = max(0, int(self.global_step) - 1)

        # safety net: always save final checkpoint at end of training.
        if self.cfg.trainer.ckpt_interval > 0:
'''


def main() -> int:
    try:
        with open(TARGET, encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"[async-final-checkpoint-step] missing {TARGET}", flush=True)
        return 1
    if MARKER in source:
        print("[async-final-checkpoint-step] already patched", flush=True)
        return 0
    range_count = source.count(OLD_RANGE)
    final_count = source.count(OLD)
    if range_count != 1 or final_count != 1:
        print(
            "[async-final-checkpoint-step] anchor mismatch; refusing: "
            f"range={range_count}, final={final_count}",
            flush=True,
        )
        return 1
    source = source.replace(OLD_RANGE, NEW_RANGE)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source.replace(OLD, NEW))
    py_compile.compile(TARGET, doraise=True)
    print(f"[async-final-checkpoint-step] patched and compiled: {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
