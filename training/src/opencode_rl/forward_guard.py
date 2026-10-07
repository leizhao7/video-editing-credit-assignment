"""Checks on the actual forward used by both stock and fused-CE workers."""
from .contract import ParityError


def configure_numerics():
    import torch
    # Pinned vLLM platforms/cuda.py disables cuDNN SDPA at import. Make the
    # same choice explicit in EVERY actual training worker, independent of
    # whether/when weight-transfer code happened to import vLLM there.
    torch.backends.cuda.enable_cudnn_sdp(False)


def check_forward(sequences, attention_mask, mm_token_type_ids, temperature):
    import torch
    if sequences.is_cuda and torch.backends.cuda.cudnn_sdp_enabled():
        raise ParityError('Training SDPA backend changed after initialization')
    if temperature != 1.0:
        raise ParityError('Actual training forward must use temperature=1')
    if mm_token_type_ids is None or mm_token_type_ids.shape != sequences.shape:
        raise ParityError('Actual forward lost recorded multimodal positions')
    if mm_token_type_ids.device != sequences.device or mm_token_type_ids.dtype != torch.long:
        raise ParityError('Multimodal positions changed device/dtype before forward')
    if attention_mask is None or attention_mask.shape != sequences.shape:
        raise ParityError('Exact multimodal training requires dense attention masks')
    if bool(((mm_token_type_ids != 0) & (mm_token_type_ids != 1)).any()):
        raise ParityError('Only actual captured image positions are supported')
    if bool(mm_token_type_ids[~attention_mask.bool()].any()):
        raise ParityError('Padding was incorrectly tagged as media')
