"""Make ``max_staleness_steps`` a hard training boundary.

SkyRL's fully-async manager only bounds aggregate producer capacity.  A slow
rollout can still finish outside the configured staleness window, and the
stock consumer merely logs a warning before training on it.  Under
``sample_full_batch`` we can safely reclassify that group as filtered, release
its producer capacity, mark its UID, and keep draining until the mini-batch is
full (or the epoch is exhausted).
"""
from __future__ import annotations

import py_compile
import sys


TARGET = "/opt/SkyRL/skyrl/train/fully_async_trainer.py"
MARKER = "S12_HARD_STALENESS_DROP"

OLD = '''        rewards = group.generator_output["rewards"]
'''

NEW = '''        # S12_HARD_STALENESS_DROP: the upstream capacity controller is
        # aggregate-only.  Never let a pathological long-running group cross
        # the configured policy-version boundary into an optimizer step.
        cur_staleness = int(self.global_step) - int(group.global_step_when_scheduled)
        if cur_staleness > int(self.max_staleness_steps):
            logger.warning(
                "Hard-dropping stale rollout group before training: "
                f"uid={group.uid}, cur_staleness={cur_staleness}, "
                f"max_staleness_steps={self.max_staleness_steps}."
            )
            return False

        # Fatal-isolation trajectories intentionally carry an all-zero loss
        # mask. GRPO needs at least two live samples to define a group-relative
        # advantage; do not let an all-dead/singleton group consume a training
        # slot merely because the stock zero-variance helper keeps <=1 live row.
        # S12_DEAD_GROUP_DROP
        loss_masks = group.generator_output["loss_masks"]
        live_count = sum(1 for mask in loss_masks if sum(mask) > 0)
        if live_count <= 1:
            logger.warning(
                "Dropping rollout group with insufficient live trajectories: "
                f"uid={group.uid}, live_count={live_count}."
            )
            return False

        rewards = group.generator_output["rewards"]
'''


def main() -> int:
    try:
        with open(TARGET, encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"[hard-staleness] missing {TARGET}", flush=True)
        return 1
    if MARKER in source:
        print("[hard-staleness] already patched", flush=True)
        return 0
    count = source.count(OLD)
    if count != 1:
        print(f"[hard-staleness] anchor matched {count} times; refusing", flush=True)
        return 1
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source.replace(OLD, NEW))
    py_compile.compile(TARGET, doraise=True)
    print(f"[hard-staleness] patched and compiled: {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
