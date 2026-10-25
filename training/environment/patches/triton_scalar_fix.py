"""Fix SkyRL's fused-linear-logprob Triton scalar specialization bug.

``num_tokens`` is included in the kernels' autotune keys, so Triton may
specialize it to a Python ``int``.  Calling ``num_tokens.to(tl.float32)`` then
fails during compilation.  Division already promotes against the fp32 tensor
numerator, so using the scalar directly is both equivalent and valid for
specialized and runtime scalar forms.
"""
from __future__ import annotations

from pathlib import Path


TARGET = Path(
    "/opt/SkyRL/skyrl/backends/skyrl_train/distributed/megatron/"
    "fused_linear_logprob_triton.py"
)


def apply(path: Path = TARGET) -> int:
    source = path.read_text()
    scalar_cast = "num_tokens.to(tl.float32)"
    ordinary_division = "/ num_tokens.to(tl.float32)"
    fused_division = ", num_tokens.to(tl.float32))"
    ordinary_count = source.count(ordinary_division)
    fused_count = source.count(fused_division)
    if ordinary_count != 2 or fused_count != 5:
        raise RuntimeError(
            "expected 2 ordinary and 5 tl.fdiv Triton scalar casts in "
            f"{path}, got ordinary={ordinary_count}, fdiv={fused_count}"
        )
    patched = source.replace(ordinary_division, "/ num_tokens")
    patched = patched.replace(
        fused_division, ", num_tokens * 1.0)"
    )
    if scalar_cast in patched:
        raise RuntimeError("Triton scalar cast replacement incomplete")
    compile(patched, str(path), "exec")
    path.write_text(patched)
    count = ordinary_count + fused_count
    print(
        f"[triton-scalar] patched {ordinary_count} ordinary + "
        f"{fused_count} fdiv num_tokens casts in {path}"
    )
    return count


if __name__ == "__main__":
    apply()
