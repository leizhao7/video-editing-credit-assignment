"""Fix left-padding positions in SkyRL's stock Qwen3.5 VLM forward path."""
from __future__ import annotations

import py_compile
import sys


TARGET = "/opt/SkyRL/skyrl/backends/skyrl_train/workers/model_wrapper.py"
MARKER = "S12_QWEN35_TEXT_POSITION_IDS"

OLD_NORMALIZE = '''            if has_image_inputs:
                # Convert TensorList -> concatenated tensors for the HF model
                if isinstance(pixel_values, TensorList):
                    pixel_values = torch.cat(pixel_values.tensors, dim=0)
                if isinstance(image_grid_thw, TensorList):
                    image_grid_thw = torch.cat(image_grid_thw.tensors, dim=0)
'''

NEW_NORMALIZE = '''            if has_image_inputs:
                # Convert TensorList -> concatenated tensors for the HF model
                if isinstance(pixel_values, TensorList):
                    pixel_values = torch.cat(pixel_values.tensors, dim=0)
                if isinstance(image_grid_thw, TensorList):
                    image_grid_thw = torch.cat(image_grid_thw.tensors, dim=0)
                # S12_QWEN35_TEXT_POSITION_IDS: an all-text microbatch is
                # represented by empty multimodal tensors.  Normalize it to
                # None so the model skips the visual tower and uses the text
                # positions already computed from attention_mask below.
                if image_grid_thw is not None and image_grid_thw.numel() == 0:
                    pixel_values, image_grid_thw = None, None
'''

OLD_CALL = '''            output = self.model(
                sequences_fwd,
                attention_mask=attention_mask_fwd,
                position_ids=None,
                **vlm_kwargs,
            )
'''

NEW_CALL = '''            output = self.model(
                sequences_fwd,
                attention_mask=attention_mask_fwd,
                # Qwen3.5 needs model-specific M-RoPE for image batches.  For
                # pure text, passing None falls back to arange(seq_len) and
                # incorrectly counts SkyRL's left PAD columns.
                position_ids=(position_ids_fwd if image_grid_thw is None else None),
                **vlm_kwargs,
            )
'''


def main() -> int:
    try:
        with open(TARGET, encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"[qwen35-position] missing {TARGET}", flush=True)
        return 1
    if MARKER in source:
        print("[qwen35-position] already patched", flush=True)
        return 0
    for old, name in ((OLD_NORMALIZE, "normalize"), (OLD_CALL, "call")):
        count = source.count(old)
        if count != 1:
            print(f"[qwen35-position] {name} anchor matched {count}; refusing", flush=True)
            return 1
    source = source.replace(OLD_NORMALIZE, NEW_NORMALIZE)
    source = source.replace(OLD_CALL, NEW_CALL)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source)
    py_compile.compile(TARGET, doraise=True)
    print(f"[qwen35-position] patched and compiled: {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
