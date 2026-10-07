"""Reuse GRPO group admission/scoring; pin the single append loop for eval too."""
from functools import partial
from opencode_rl.contract import ParityError
from opencode_rl.train_generator import OpenCodeGenerator
from opencode_rl.training import build_packed_generator_output, prepare_trajectory
from .contract import validate_runtime
from .task_rollout import rollout_task


class AppendGenerator(OpenCodeGenerator):
    def __init__(self, **kwargs):
        if any(key in kwargs for key in ('rollout', 'build_output', 'prepare', 'preparation_policy')):
            raise ParityError('Train/eval cannot substitute a different append agent loop')
        validate_runtime(kwargs['config'], kwargs['contract'])
        enabled = kwargs.get('process_judge') is not None
        super().__init__(**kwargs, rollout=rollout_task,
            build_output=(partial(build_append_output, enable_packed_process=True)
                          if enabled else build_append_output),
            prepare=partial(prepare_trajectory, deduplicate_tensors=True), preparation_policy='scored_first')


def build_append_output(store, attempts, contract, **kwargs):
    output = build_packed_generator_output(store, attempts, contract, **kwargs)
    if len(output['response_ids']) != len(attempts):
        raise ParityError('Append training requires exactly one packet per complete trajectory')
    return output
