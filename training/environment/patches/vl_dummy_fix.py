"""Keep Qwen3.5 VLM FSDP collectives aligned on text-only microbatches.

Qwen3.5 conditionally executes ``model.visual``.  With layer-wise FSDP2, an
image rank therefore enters visual parameter all-gathers while a text-only
rank skips them.  The original workaround called ``get_image_features`` before
the Hugging Face model forward on text-only ranks.  That was subtly wrong:

* image rank:     embedding all-gather -> visual all-gathers -> language
* text-only rank: visual all-gathers   -> embedding all-gather -> language

Collectives with compatible sizes could silently consume one another and feed
the wrong parameters to a layer (the Step-7 ~9-11 nat logprob split); a later
shape/order difference then stalled until the NCCL watchdog aborted the job.

This patch sends text-only inputs through the *ordinary* Hugging Face image
branch instead.  It appends one minimal image placeholder per sequence, marks
that position as attention padding, supplies matching zero pixels, and removes
the placeholder hidden state before returning.  Putting the dummy at the end is
important: all real token indices and causal prefixes remain byte-for-byte
unchanged.  Consequently every rank uses the same embedding -> visual ->
language collective order, while output length, loss mask and trajectory
contents are unchanged.

The production recipe freezes the vision tower before FSDP/optimizer setup.
This patch deliberately fails if a visual parameter is trainable: the masked
dummy has no visual loss path, so using it with trainable vision weights would
reintroduce backward-collective skew.
"""

from __future__ import annotations

import torch


def _visual_is_frozen(model) -> bool:
    visual = getattr(model, "visual", None)
    if visual is None:
        return False
    params = list(visual.parameters())
    return bool(params) and not any(param.requires_grad for param in params)


def _append_masked_dummy(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.Tensor | None,
):
    """Build the minimal valid image input without changing real-token context."""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "VL dummy requires 2-D input_ids and a same-shape attention_mask, "
            f"got ids={tuple(input_ids.shape)} mask={tuple(attention_mask.shape)}"
        )
    if not _visual_is_frozen(model):
        raise RuntimeError(
            "VL dummy requires a completely frozen vision tower; otherwise "
            "text-only ranks would skip visual backward collectives"
        )

    batch = int(input_ids.shape[0])
    if batch <= 0:
        raise ValueError("VL dummy received an empty batch")
    vc = model.config.vision_config
    merge = int(getattr(vc, "spatial_merge_size", 2))
    temporal = int(getattr(vc, "temporal_patch_size", 1))
    patch = int(getattr(vc, "patch_size", 1))
    channels = int(getattr(vc, "in_channels", 3))
    if merge <= 0 or temporal <= 0 or patch <= 0 or channels <= 0:
        raise ValueError("invalid Qwen3.5 vision patch configuration")

    device = input_ids.device
    dummy_ids = torch.full(
        (batch, 1),
        int(model.config.image_token_id),
        dtype=input_ids.dtype,
        device=device,
    )
    padded_ids = torch.cat((input_ids, dummy_ids), dim=1)
    padded_mask = torch.cat(
        (attention_mask, torch.zeros((batch, 1), dtype=attention_mask.dtype, device=device)),
        dim=1,
    )

    if mm_token_type_ids is None:
        mm_token_type_ids = torch.zeros_like(input_ids)
    elif mm_token_type_ids.shape != input_ids.shape:
        raise ValueError(
            "VL dummy mm_token_type_ids shape mismatch: "
            f"types={tuple(mm_token_type_ids.shape)} ids={tuple(input_ids.shape)}"
        )
    padded_types = torch.cat(
        (
            mm_token_type_ids,
            torch.ones((batch, 1), dtype=mm_token_type_ids.dtype, device=device),
        ),
        dim=1,
    )

    # grid [1, merge, merge] yields exactly one post-merge image embedding per
    # sequence, matching the one appended placeholder token.
    grid = torch.tensor(
        [[1, merge, merge]], dtype=torch.long, device=device
    ).repeat(batch, 1)
    raw_patches_per_image = merge * merge
    patch_dim = channels * temporal * patch * patch
    pixels = torch.zeros(
        (batch * raw_patches_per_image, patch_dim),
        dtype=model.visual.dtype,
        device=device,
    )
    return padded_ids, padded_mask, padded_types, pixels, grid


def _remove_dummy_output(outputs, expected_length: int):
    hidden = getattr(outputs, "last_hidden_state", None)
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(
            "Qwen3_5Model VL dummy requires a ModelOutput with last_hidden_state"
        )
    if hidden.ndim != 3 or int(hidden.shape[1]) != expected_length + 1:
        raise RuntimeError(
            "VL dummy output length drift: "
            f"hidden={tuple(hidden.shape)} expected_sequence={expected_length + 1}"
        )
    # Slicing off the masked suffix leaves a non-contiguous view (the row
    # stride still includes the removed token).  SkyRL's fused linear-CE
    # kernel flattens this tensor with ``view`` and therefore requires a
    # contiguous layout.  Restore the ordinary Qwen forward contract here,
    # at the point where the dummy implementation creates the view.
    outputs.last_hidden_state = hidden[:, :-1, :].contiguous()
    return outputs


def _append_masked_position(position_ids: torch.Tensor, input_ids: torch.Tensor):
    """Preserve SkyRL's real-token text positions and add a masked suffix."""
    if position_ids.ndim == 2 and tuple(position_ids.shape) == tuple(input_ids.shape):
        suffix_shape = (int(position_ids.shape[0]), 1)
    elif (
        position_ids.ndim == 3
        and int(position_ids.shape[-2]) == int(input_ids.shape[0])
        and int(position_ids.shape[-1]) == int(input_ids.shape[1])
    ):
        suffix_shape = (*position_ids.shape[:-1], 1)
    else:
        raise ValueError(
            "VL dummy position_ids shape mismatch: "
            f"positions={tuple(position_ids.shape)} ids={tuple(input_ids.shape)}"
        )
    # This value is never attended to or returned.  Reusing the final real
    # position avoids extending a multimodal RoPE axis unnecessarily.
    suffix = position_ids[..., -1:].clone().reshape(suffix_shape)
    return torch.cat((position_ids, suffix), dim=-1)


def apply() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    cls = m.Qwen3_5Model
    if getattr(cls, "_vl_dummy_patched", False):
        return
    original_forward = cls.forward

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        use_dummy = (
            pixel_values is None
            and pixel_values_videos is None
            and image_grid_thw is None
            and video_grid_thw is None
            and input_ids is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        original_length = None
        if use_dummy:
            if inputs_embeds is not None:
                raise RuntimeError("VL dummy does not support simultaneous input_ids/inputs_embeds")
            if attention_mask is None:
                raise RuntimeError("VL dummy requires attention_mask for the masked placeholder")
            if past_key_values is not None:
                raise RuntimeError("VL dummy is training-only and does not support KV cache")
            original_length = int(input_ids.shape[1])
            if position_ids is not None:
                position_ids = _append_masked_position(position_ids, input_ids)
            (
                input_ids,
                attention_mask,
                mm_token_type_ids,
                pixel_values,
                image_grid_thw,
            ) = _append_masked_dummy(
                self, input_ids, attention_mask, mm_token_type_ids
            )
            if not getattr(cls, "_vl_dummy_logged", False):
                cls._vl_dummy_logged = True
                print(
                    "[VL_DUMMY_ORDERED] masked image path enabled "
                    f"(training={self.training}, grad={torch.is_grad_enabled()})",
                    flush=True,
                )

        outputs = original_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        if original_length is not None:
            outputs = _remove_dummy_output(outputs, original_length)
        return outputs

    cls.forward = forward
    cls._vl_dummy_original_forward = original_forward
    cls._vl_dummy_patched = True
    print("[VL_DUMMY_ORDERED] Qwen3_5Model.forward wrapped", flush=True)


def self_test() -> None:
    """CPU tensor-contract checks for suffix construction and output slicing."""
    from types import SimpleNamespace

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        @property
        def dtype(self):
            return self.weight.dtype

    model = SimpleNamespace(
        visual=Visual(),
        config=SimpleNamespace(
            image_token_id=63,
            vision_config=SimpleNamespace(
                spatial_merge_size=2,
                temporal_patch_size=1,
                patch_size=2,
                in_channels=3,
            ),
        ),
    )
    ids = torch.tensor([[4, 5, 6], [7, 8, 9]])
    mask = torch.ones_like(ids)
    padded_ids, padded_mask, types, pixels, grid = _append_masked_dummy(
        model, ids, mask, None
    )
    assert padded_ids.tolist() == [[4, 5, 6, 63], [7, 8, 9, 63]]
    assert padded_mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 0]]
    assert types.tolist() == [[0, 0, 0, 1], [0, 0, 0, 1]]
    assert tuple(pixels.shape) == (8, 12)
    assert grid.tolist() == [[1, 2, 2], [1, 2, 2]]
    output = SimpleNamespace(last_hidden_state=torch.randn(2, 4, 5))
    sliced = _remove_dummy_output(output, 3)
    assert tuple(sliced.last_hidden_state.shape) == (2, 3, 5)
    assert sliced.last_hidden_state.is_contiguous()
    positions = torch.tensor([[0, 1, 2], [0, 1, 2]])
    assert _append_masked_position(positions, ids).tolist() == [
        [0, 1, 2, 2], [0, 1, 2, 2]
    ]


def integration_self_test() -> None:
    """Compare real tiny-Qwen outputs with and without the ordered dummy path."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    apply()
    cfg = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": ["full_attention"],
            "max_position_embeddings": 64,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    model = Qwen3_5Model(cfg).eval()
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.ones_like(ids)

    old_available = torch.distributed.is_available
    old_initialized = torch.distributed.is_initialized
    try:
        torch.distributed.is_available = lambda: True
        torch.distributed.is_initialized = lambda: False
        with torch.no_grad():
            baseline = model(
                ids, attention_mask=mask, use_cache=False
            ).last_hidden_state
        torch.distributed.is_initialized = lambda: True
        with torch.no_grad():
            actual = model(
                ids, attention_mask=mask, use_cache=False
            ).last_hidden_state
    finally:
        torch.distributed.is_available = old_available
        torch.distributed.is_initialized = old_initialized

    assert tuple(actual.shape) == tuple(baseline.shape)
    torch.testing.assert_close(actual, baseline, rtol=1e-5, atol=1e-6)
    assert all(parameter.grad is None for parameter in model.visual.parameters())
