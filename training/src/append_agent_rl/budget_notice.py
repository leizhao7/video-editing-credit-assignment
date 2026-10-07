"""Public harness budget facts, not tool observations or reward evidence.

Installed as append_agent_rl.budget_notice only in newly qualified images.
Notices never remove history, change a limit, stop a generation, or add a score.
"""
from copy import deepcopy
import hashlib

VERSION = 'append-shared-delivery-closeout-v1'
FINAL_DELIVERY_PROMPT = "FINAL DELIVERY TURN — this is the last model turn for this task.\nThe existing tools remain available under the same sandbox permissions. If a required\ndeliverable is missing or unfinished, use this turn to save or finish it with the\navailable tools. Prioritize the task's required output files over a prose summary.\nDo not begin more optional inspection or refinement. Do not invent an answer or\nclaim that files were saved or rendered without execution evidence. If the files\nare already complete, finish briefly. Tool calls on this turn will execute, but\nthere will be no further model turn afterward; do not rely on a future correction.\nA JSON block in a chat reply is not a replacement for the required JSON file.\nFor S4E, an edit plan alone is not the required rendered output/preview.mp4.\nThis reminder does not relax the task requirements or grant additional permissions."


def _limits(limits):
    values = {k: getattr(limits, k) for k in ('context', 'output', 'turns')}
    if (any(type(v) is not int or v < 1 for v in values.values())
            or values['output'] >= values['context']):
        raise ValueError('Invalid actual harness limits')
    return values


def initial_profile(profile, limits):
    values = _limits(limits)
    if (type(profile) is not dict or not isinstance(profile.get('task'), str)
            or not profile['task'].strip() or 'harness_budget' in profile
            or not isinstance(profile.get('final_turn_prompt'), str)
            or not profile['final_turn_prompt'].strip()):
        raise ValueError('Missing/or already budget-bound native profile')
    context, output, turns = (values[k] for k in ('context', 'output', 'turns'))
    ending = (f'Turn {turns} remains available for final delivery tool calls. '
        f'Save a first workable delivery earlier, then verify and refine it. '
        f'Do not postpone all file writes or rendering to the final turn: '
        f'no subsequent model turn will be available to repair a failed command. '
        f'A prose summary or chat JSON block is not a substitute for the required files.'
        if turns > 1 else 'There is only one model turn; any tool action must fit in that turn.')
    text = (f'HARNESS BUDGET — applies to this task from the start:\n'
        f'At most {turns} model turns, with at most {output} generated tokens per turn. '
        f'The cumulative context limit is {context} tokens, including the initial '
        f'prompt, reasoning, replies, tool results, and image tokens. '
        f'The context budget may end the task before the turn limit. {ending}\n'
        f'Plan inspection and editing within these limits. Once evidence supports '
        f'a workable task-specific deliverable, write it and then refine it as '
        f'needed. Finish early when the task is complete; do not keep inspecting '
        f'only to consume the remaining budget. Do not guess unsupported content '
        f'or treat this budget notice as evidence that a file exists or is correct.')
    updated = deepcopy(profile)
    updated['final_turn_prompt'] = FINAL_DELIVERY_PROMPT
    updated['task'] = profile['task'] + '\n\n' + text
    updated['harness_budget'] = {'version': VERSION, **values,
        'original_final_turn_prompt_sha256': hashlib.sha256(profile['final_turn_prompt'].encode()).hexdigest(),
        'effective_final_turn_prompt_sha256': hashlib.sha256(FINAL_DELIVERY_PROMPT.encode()).hexdigest(),
        'original_task_sha256': hashlib.sha256(profile['task'].encode()).hexdigest(),
        'effective_task_sha256': hashlib.sha256(updated['task'].encode()).hexdigest()}
    return updated


def checked_notices(notices, *, end_turn):
    """Only caller-supplied notes already shown before a completed action.

    Later note bodies are not inspected, copied, hashed or sent to the judge.
    These records deliberately supply no execution/media/score evidence IDs.
    """
    if type(end_turn) is not int or end_turn < 0 or type(notices) not in (list, tuple) or len(notices) > 256:
        raise ValueError('Invalid bounded harness-notice prefix')
    result, previous = [], 0
    for notice in notices:
        if type(notice) is not dict or type(notice.get('before_turn')) is not int or notice['before_turn'] < 1:
            raise ValueError('Invalid harness notice boundary')
        n = notice['before_turn']
        if n > end_turn:
            continue
        if (set(notice) != {'before_turn', 'text'} or n <= previous
                or not isinstance(notice['text'], str) or not 0 < len(notice['text']) <= 12000):
            raise ValueError('Invalid/duplicate/out-of-order visible harness notice')
        result.append(deepcopy(notice))
        previous = n
    return result


class BudgetNotices:
    def __init__(self, limits, final_turn_prompt):
        self.limits = _limits(limits)
        if not isinstance(final_turn_prompt, str) or not 0 < len(final_turn_prompt) <= 12000:
            raise ValueError('Missing bounded original final-turn instruction')
        self.final_turn_prompt = final_turn_prompt
        self.last_turn = self.last_tokens = 0
        self.turn_band = self.context_band = 0

    def after_tool(self, *, completed_turn, tokens_used):
        turns, context = self.limits['turns'], self.limits['context']
        if (type(completed_turn) is not int or type(tokens_used) is not int
                or not self.last_turn < completed_turn < turns
                or not self.last_tokens <= tokens_used < context):
            raise ValueError('Non-monotone/finished harness budget observation')
        self.last_turn, self.last_tokens = completed_turn, tokens_used
        if completed_turn + 1 == turns:
            # Return the controller-bound delivery instruction, not OpenCode's text-only default.
            return {'before_turn': turns, 'text': self.final_turn_prompt}
        turn_band = max((p for p in (50, 80, 95) if completed_turn * 100 >= turns * p), default=0)
        context_band = max((p for p in (70, 85, 95) if tokens_used * 100 >= context * p), default=0)
        changed = turn_band > self.turn_band or context_band > self.context_band
        self.turn_band, self.context_band = max(turn_band, self.turn_band), max(context_band, self.context_band)
        if not changed:
            return None
        return {'before_turn': completed_turn + 1, 'text':
            f'HARNESS BUDGET UPDATE: {completed_turn}/{turns} model turns completed. '
            f'Turn {turns} permits final delivery tool calls, but no subsequent model correction. '
            f'{tokens_used}/{context} context tokens were already used BEFORE the '
            f'latest tool observation and this notice; actual remaining space is '
            f'less than {context-tokens_used}. Save/update the required task files '
            f'while tools and context remain. If the result is ready, finish now. '
            f'This notice does not establish file existence, quality, or task success.'}
