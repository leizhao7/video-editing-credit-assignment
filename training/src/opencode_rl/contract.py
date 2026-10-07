"""Hash-pinned execution contract shared by rollout, evaluation and training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

OPENCODE_COMMIT = "d7a6e1daaf2271bcd0b611fbb27d4956806e475a"
SKYRL_COMMIT = "59d4daedee24c7b1a79d857bbf322a6d195c3792"
VLLM_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
HASH = re.compile(r"^[a-f0-9]{64}$")
ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


class ParityError(ValueError):
    """A sample cannot be proven to match the actual inference request."""


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Contract:
    binary_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    processor_sha256: str
    tools_sha256: str
    sandbox_sha256: str
    integration_sha256: str
    max_context: int
    max_output: int
    thinking: bool
    temperature: float
    top_p: float
    top_k: int
    image_token_id: int
    vision_patch_dim: int
    spatial_merge_size: int
    opencode_commit: str = OPENCODE_COMMIT
    skyrl_commit: str = SKYRL_COMMIT
    vllm_commit: str = VLLM_COMMIT
    model_revision: str = MODEL_REVISION
    schema: str = "opencode-rl-contract-v1"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.endswith("_sha256") and not HASH.fullmatch(value):
                raise ParityError(f"Invalid component digest: {key}")
        if (self.opencode_commit, self.skyrl_commit, self.vllm_commit, self.model_revision) != (
            OPENCODE_COMMIT, SKYRL_COMMIT, VLLM_COMMIT, MODEL_REVISION
        ):
            raise ParityError("Unreviewed source/model version")
        if not 0 < self.max_output < self.max_context or type(self.thinking) is not bool:
            raise ParityError("Invalid context/output/thinking configuration")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ParityError("RL requires explicit positive temperature and valid top_p")
        if self.temperature != 1.0:
            raise ParityError("This training recipe requires temperature=1 for raw policy logprobs")
        if self.image_token_id < 0 or self.vision_patch_dim <= 0 or self.spatial_merge_size <= 0:
            raise ParityError("Missing model-specific multimodal shape contract")

    @property
    def sha256(self):
        return digest(asdict(self))

    def to_dict(self):
        return asdict(self)


def require_ids(values, label):
    if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values):
        raise ParityError(f"Missing/invalid {label}")
