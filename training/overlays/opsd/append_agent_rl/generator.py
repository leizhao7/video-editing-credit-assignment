"""Reuse GRPO group admission/scoring; pin the single append loop for eval too."""
from opencode_rl.contract import ParityError
from opencode_rl.train_generator import OpenCodeGenerator
from opencode_rl.training import build_packed_generator_output
from .contract import validate_runtime
from .task_rollout import rollout_task


class AppendGenerator(OpenCodeGenerator):
    def __init__(self, **kwargs):
        if 'rollout' in kwargs or 'build_output' in kwargs:
            raise ParityError('Train/eval cannot substitute a different append agent loop')
        validate_runtime(kwargs['config'], kwargs['contract'])
        super().__init__(**kwargs, rollout=rollout_task, build_output=build_append_output)


def build_append_output(store, attempts, contract, **kwargs):
    output = build_packed_generator_output(store, attempts, contract, **kwargs)
    if len(output['response_ids']) != len(attempts):
        raise ParityError('Append training requires exactly one packet per complete trajectory')
    return output
