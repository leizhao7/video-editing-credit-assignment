"""Evaluation identities do NOT assert numerical training parity for new models."""
from pathlib import Path
import hashlib
import json

from .assets import files_hash
from .contract import Contract, HASH, OPENCODE_COMMIT, ParityError, digest
from .azure_gpt55_transport import AZURE_GPT55_MODEL

MODEL_27_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


class EvaluationContract:
    def __init__(self, *, components, model, settings, schema="opencode-evaluation-contract-v1"):
        if schema != "opencode-evaluation-contract-v1":
            raise ParityError("Unreviewed evaluation contract")
        if set(components) != {"binary_sha256", "config_sha256", "tools_sha256", "sandbox_sha256", "integration_sha256"}:
            raise ParityError("Incomplete runtime components")
        if any(not isinstance(v, str) or not HASH.fullmatch(v) for v in components.values()):
            raise ParityError("Invalid runtime hashes")
        if model.get("provider") == "vllm":
            if (model.get("model_id"), model.get("model_revision")) != ("Qwen/Qwen3.6-27B", MODEL_27_REVISION):
                raise ParityError("Unpinned GPU evaluation model")
            for key in ("tokenizer_sha256", "template_sha256", "processor_sha256"):
                if not HASH.fullmatch(str(model.get(key, ""))):
                    raise ParityError("Missing actual GPU model assets")
            if any(type(model.get(k)) is not int or model[k] <= 0 for k in
                   ("image_token_id", "vision_patch_dim", "spatial_merge_size")):
                raise ParityError("Missing GPU visual shapes")
        elif model not in ({"provider": "google-openai-compatible", "model_id": "gemini-3.8-flash",
                       "model_revision": "API-managed; actual response model recorded", "internal_tensors_available": False},
                       AZURE_GPT55_MODEL):
            raise ParityError("Unreviewed API backend")
        if settings != {"max_context": 131072, "max_output": 16384, "thinking": True,
                        "temperature": 1.0, "top_p": 1.0, "top_k": -1}:
            raise ParityError("Unexpected shared evaluation recipe")
        self.components, self.model, self.settings = components, model, settings

    def __getattr__(self, name):
        for mapping in (self.components, self.model, self.settings):
            if name in mapping:
                return mapping[name]
        raise AttributeError(name)

    def to_dict(self):
        return {"schema": "opencode-evaluation-contract-v1", "components": self.components,
                "model": self.model, "settings": self.settings}

    @property
    def sha256(self):
        return digest(self.to_dict())


def load_contract(data):
    return EvaluationContract(**data) if data.get("schema") == "opencode-evaluation-contract-v1" else Contract(**data)


def evaluation_assets(*, package, config, sandbox_id, tools_sha256, model_dir=None):
    package = Path(package)
    components = {"binary_sha256": hashlib.sha256(Path("/usr/local/bin/opencode").read_bytes()).hexdigest(),
        "config_sha256": digest(config), "sandbox_sha256": digest({"modal_image_id": sandbox_id}),
        "tools_sha256": tools_sha256,
        "integration_sha256": files_hash(package, [p.name for p in package.iterdir() if p.suffix in (".py", ".mjs")])}
    model = {"provider": "google-openai-compatible", "model_id": "gemini-3.8-flash",
             "model_revision": "API-managed; actual response model recorded", "internal_tensors_available": False}
    if config.get('model') == 'ocrl/gpt-5.5':
        if model_dir is not None:
            raise ParityError('API policy cannot use a GPU model directory')
        model = dict(AZURE_GPT55_MODEL)
    if model_dir:
        root = Path(model_dir)
        if root.name != MODEL_27_REVISION:
            raise ParityError("Unexpected 27B snapshot")
        cfg = json.loads((root / "config.json").read_bytes())
        vision = cfg["vision_config"]
        model = {"provider": "vllm", "model_id": "Qwen/Qwen3.6-27B", "model_revision": root.name,
            "tokenizer_sha256": files_hash(root, ["tokenizer.json", "tokenizer_config.json"]),
            "template_sha256": files_hash(root, [p.name for p in root.iterdir() if "chat_template" in p.name] or ["tokenizer_config.json"]),
            "processor_sha256": files_hash(root, [p.name for p in root.iterdir() if p.name in
                ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "config.json")]),
            "image_token_id": cfg["image_token_id"],
            "vision_patch_dim": vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2,
            "spatial_merge_size": vision["spatial_merge_size"]}
    return EvaluationContract(components=components, model=model, settings={"max_context": 131072,
        "max_output": 16384, "thinking": True, "temperature": 1.0, "top_p": 1.0, "top_k": -1})
