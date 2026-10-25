"""Freeze Qwen3.5's vision tower before FSDP and optimizer construction.

The v8 9B policy was full-fine-tuning ``model.visual``.  Exact replays of the
failed step-5 batch proved that the PPO/CE boundary gradients are finite and
that the first non-finite gradients appear inside the vision tower.  Training
the pretrained visual encoder is unnecessary for this run: the policy still
sees exactly the same rendered frames and visual features, while only the
language side is updated by RL.

This patch wraps ``HFModelWrapper.__init__`` because that is the last stable
point after the Hugging Face VLM exists and before SkyRL applies FSDP2 and
constructs AdamW.  ``requires_grad=False`` therefore prevents both visual
backward kernels and visual gradient collectives.  The visual parameters stay
in the model/state_dict and are still sent to inference engines unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path


FSDP_WORKER = Path(
    "/opt/SkyRL/skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py"
)


def _visual_module(wrapper):
    model = getattr(wrapper, "model", None)
    inner = getattr(model, "model", None)
    visual = getattr(inner, "visual", None)
    if visual is None:
        visual = getattr(model, "visual", None)
    return visual


def assert_frozen(wrapper) -> tuple[int, int]:
    """Return ``(parameter_tensors, elements)`` or fail on a partial freeze."""
    visual = _visual_module(wrapper)
    if visual is None:
        return (0, 0)
    params = list(visual.parameters())
    trainable = [p for p in params if p.requires_grad]
    if trainable:
        raise RuntimeError(
            f"vision freeze incomplete: {len(trainable)}/{len(params)} "
            "parameter tensors still require gradients"
        )
    return len(params), sum(int(p.numel()) for p in params)


def freeze_wrapper(wrapper) -> tuple[int, int]:
    """Freeze one already-constructed HFModelWrapper, before FSDP wrapping."""
    visual = _visual_module(wrapper)
    if visual is None:
        return (0, 0)
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    tensors, elements = assert_frozen(wrapper)
    print(
        "[VISION_FROZEN] "
        f"pid={os.getpid()} tensors={tensors} elements={elements}",
        flush=True,
    )
    return tensors, elements


def apply() -> bool:
    if os.environ.get("SKYRL_FREEZE_VISION", "0") != "1":
        return False

    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

    if getattr(HFModelWrapper, "_s12_freeze_vision_patched", False):
        return False
    original_init = HFModelWrapper.__init__

    def frozen_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        freeze_wrapper(self)

    HFModelWrapper.__init__ = frozen_init
    HFModelWrapper._s12_freeze_vision_patched = True
    HFModelWrapper._s12_stock_init_before_vision_freeze = original_init
    print("[freeze_vision] HFModelWrapper pre-FSDP freeze installed", flush=True)
    return True


def patch_fsdp_worker(path: Path = FSDP_WORKER) -> bool:
    """Install a source-level, policy-only freeze as a deterministic backstop.

    Ray may import Python modules during worker bootstrap before a job runtime
    environment is fully attached.  The runtime monkey patch above remains a
    useful early path, but production correctness must not depend on that
    timing.  This build-time insertion executes directly in
    ``FSDPPolicyWorkerBase.init_model`` after construction and before FSDP2 or
    AdamW, and fails fast if the expected SkyRL source moved.
    """
    source = path.read_text()
    marker = "# S12_VISION_FREEZE_SOURCE_PATCH"
    if marker in source:
        return False
    class_needle = "class FSDPPolicyWorkerBase"
    class_start = source.find(class_needle)
    if class_start < 0:
        raise RuntimeError(f"policy worker class not found in {path}")
    class_end = source.find("\nclass ", class_start + len(class_needle))
    if class_end < 0:
        class_end = len(source)
    policy_source = source[class_start:class_end]
    needle = "        self._seq_parallel_monkey_patch(model=wrapped_model.model)\n"
    if policy_source.count(needle) != 1:
        raise RuntimeError(
            f"expected one policy pre-FSDP insertion point in {path}, "
            f"got {policy_source.count(needle)}"
        )
    replacement = (
        "        # S12_VISION_FREEZE_SOURCE_PATCH\n"
        "        if os.environ.get('SKYRL_FREEZE_VISION', '0') == '1':\n"
        "            from freeze_vision import freeze_wrapper\n"
        "            _n_v, _e_v = freeze_wrapper(wrapped_model)\n"
        "            if _n_v <= 0 or _e_v <= 0:\n"
        "                raise RuntimeError('SKYRL_FREEZE_VISION=1 but no visual parameters found')\n"
        + needle
    )
    patched_policy = policy_source.replace(needle, replacement, 1)
    path.write_text(source[:class_start] + patched_policy + source[class_end:])
    compile(path.read_text(), str(path), "exec")
    print(f"[freeze_vision] source patch installed: {path}")
    return True


if __name__ == "__main__":
    patch_fsdp_worker()
