"""Distinct harness identity. Never relabel an old native-loop checkpoint."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

from opencode_rl.contract import Contract, HASH, ParityError, digest
from opencode_rl.runtime import configuration as native_configuration


def loop_source_sha256():
    root = Path(__file__).parent
    files = sorted(p for p in root.rglob('*') if p.is_file() and
                   p.suffix in ('.py', '.ts') and 'tests' not in p.parts and
                   not p.name.endswith('.test.ts'))
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


@dataclass(frozen=True)
class AppendContract(Contract):
    schema: str = 'append-opencode-rl-contract-v1'
    loop_sha256: str = ''
    max_turns: int = 100

    def __post_init__(self):
        super().__post_init__()
        if self.schema != 'append-opencode-rl-contract-v1' or not HASH.fullmatch(self.loop_sha256):
            raise ParityError('A distinct append-loop source identity is required')
        if self.top_p != 1.0 or self.top_k != -1 or type(self.max_turns) is not int or self.max_turns < 1:
            raise ParityError('Unreviewed append-loop sampling/turn contract')


def configuration(**kwargs):
    config = native_configuration(**kwargs)
    # Keep the same tools/provider/system selection. No native compactor may
    # rewrite historical sampled IDs in this harness.
    config['compaction'] = {'auto': False, 'prune': False, 'tail_turns': 2}
    return config


def validate_runtime(config, contract):
    if not isinstance(contract, AppendContract) or digest(config) != contract.config_sha256:
        raise ParityError('Wrong harness/config identity')
    if contract.loop_sha256 != loop_source_sha256():
        raise ParityError('Append-loop sources changed after contract creation')
    if config.get('compaction', {}).get('auto') is not False or config['compaction'].get('prune') is not False:
        raise ParityError('History rewriting is disabled in the append-only harness')
    agent = config.get('agent', {}).get('build', {})
    if (config.get('model') != 'ocrl/qwen35-9b-policy' or agent.get('steps') != contract.max_turns or
            agent.get('temperature') != contract.temperature or agent.get('top_p') != contract.top_p):
        raise ParityError('Tool host model/turn/sampling profile differs from the controller')
    model = config.get('provider', {}).get('ocrl', {}).get('models', {}).get('qwen35-9b-policy', {})
    if (model.get('reasoning') is not contract.thinking or
            model.get('limit') != {'context': contract.max_context, 'output': contract.max_output}):
        raise ParityError('Tool host context/output/thinking declaration differs from inference')
