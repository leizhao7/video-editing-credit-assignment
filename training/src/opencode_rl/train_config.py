"""Concrete dataclass extension, importable without starting Ray or inference."""
from dataclasses import dataclass, field


def config_class(base=None, generator=None):
    if base is None:
        from skyrl.train.config import SkyRLTrainConfig, GeneratorConfig
        base, generator = SkyRLTrainConfig, GeneratorConfig

    @dataclass
    class OpenCodeGeneratorConfig(generator):
        ocrl_plan: str = ''
        ocrl_process_weight: float = 0.0
        ocrl_contribution_dir: str = ''
        ocrl_causal_prefix: bool = False
        ocrl_packed_prm_enabled: bool = False
        ocrl_gradient_probe_interval: int = 0

    @dataclass
    class OpenCodeConfig(base):
        generator: OpenCodeGeneratorConfig = field(default_factory=OpenCodeGeneratorConfig)

    return OpenCodeConfig
