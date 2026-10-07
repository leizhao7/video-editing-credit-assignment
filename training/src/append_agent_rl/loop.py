"""One shared append-only state machine; no model/tool implementation copies.

`engine.generate` must return its independently captured engine-admission turn.
`renderer.observation` may render ONLY newly returned tool observations, never
an assistant message or the historical conversation. All callbacks are async.
"""
from __future__ import annotations

from .copying import deepcopy
from dataclasses import dataclass
import math

from opencode_rl.causal_prefix import can_extend, plan_chains
from opencode_rl.contract import HASH, ID, ParityError, digest, require_ids
from opencode_rl.visual_budget import MAX_CONTEXT_IMAGES


@dataclass(frozen=True)
class Limits:
    context: int = 131072
    output: int = 16384
    turns: int = 100

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in (self.context, self.output, self.turns))
                or self.output >= self.context):
            raise ParityError('Invalid append-loop limits')


def checked_segment(segment):
    """Renderer media offsets are relative to this segment, not old history."""
    segment = deepcopy(segment)
    if set(segment) != {'token_ids', 'features'}:
        raise ParityError('Unexpected rendered segment fields')
    require_ids(segment['token_ids'], 'rendered segment IDs')
    if type(segment['features']) is not list:
        raise ParityError('Missing rendered feature inventory')
    previous = 0
    for feature in segment['features']:
        lo, n = feature.get('offset'), feature.get('length')
        if (feature.get('modality') != 'image' or type(lo) is not int or type(n) is not int
                or n < 1 or not previous <= lo < lo+n <= len(segment['token_ids'])
                or not feature.get('data')):
            raise ParityError('Invalid image positions/provenance in new segment')
        previous = lo+n
    return segment


class AppendState:
    def __init__(self, initial, *, attempt_id, session_id, contract_sha256, limits):
        if not ID.fullmatch(attempt_id) or not ID.fullmatch(session_id) or not HASH.fullmatch(contract_sha256):
            raise ParityError('Invalid append-loop identity')
        self.limits = limits
        self.identity = {'attempt_id': attempt_id, 'session_id': session_id,
                         'contract_sha256': contract_sha256}
        segment = checked_segment(initial)
        if len(segment['token_ids']) >= limits.context or len(segment['features']) > MAX_CONTEXT_IMAGES:
            raise ParityError('Initial public prompt exceeds context')
        self._ids, self._features = segment['token_ids'], segment['features']
        self._turns = []
        self._phase = 'ready'
        self.reason = None
        self._pending = None

    @property
    def turns(self):
        return deepcopy(self._turns)

    @property
    def turn_count(self):
        return len(self._turns)

    @property
    def request_ids(self):
        return [t['request_id'] for t in self._turns]

    def request(self):
        if self._phase != 'ready' or self.reason:
            raise ParityError('Generation outside a ready append-only state')
        remaining = self.limits.context - len(self._ids)
        if remaining <= 0 or len(self._turns) >= self.limits.turns:
            raise ParityError('Generation after an exhausted loop budget')
        self._pending = {'prompt_token_ids': deepcopy(self._ids), 'features': deepcopy(self._features),
                         'max_tokens': min(self.limits.output, remaining),
                         'metadata': {**self.identity, 'agent': 'build', 'primary': True,
                                      'turn': len(self._turns)}}
        self._phase = 'generating'
        return deepcopy(self._pending)

    def accept(self, turn):
        if self._phase != 'generating' or self._pending is None:
            raise ParityError('Duplicate/unsolicited engine completion')
        turn = deepcopy(turn)
        expected, actual = self._pending, turn['input']
        # Evidence is produced at the engine boundary, not by the caller
        # echoing its intended request into a fabricated success record.
        for key in ('prompt_token_ids', 'features'):
            if actual[key] != expected[key]:
                raise ParityError(f'Actual engine input changed: {key}')
        if any(turn['metadata'].get(k) != v for k, v in expected['metadata'].items()):
            raise ParityError('Engine completion crossed a loop identity/turn boundary')
        if actual.get('contract_sha256') != self.identity['contract_sha256'] or not actual.get('weight_version'):
            raise ParityError('Missing actual engine contract/weight identity')
        payload = {k: v for k, v in actual.items() if k != 'input_sha256'}
        if actual.get('input_sha256') != digest(payload):
            raise ParityError('Actual engine input digest changed')
        if actual.get('logprobs_mode') != 'raw_logprobs':
            raise ParityError('Behavior probabilities must be raw engine logprobs')
        sampling = actual.get('sampling', {})
        if (sampling.get('max_tokens') != expected['max_tokens'] or sampling.get('n') != 1
                or sampling.get('temperature') != 1.0 or sampling.get('top_p') != 1.0
                or sampling.get('top_k') != -1):
            raise ParityError('Engine sampling differs from shared T1 contract')
        require_ids(turn['response_ids'], 'actual generated IDs')
        n = len(turn['response_ids'])
        if (n > expected['max_tokens'] or len(turn['rollout_logprobs']) != n
                or turn['loss_mask'] != [1]*n or turn['finish_reason'] not in ('stop', 'length')):
            raise ParityError('Incomplete/raw action alignment or termination')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= -9999 or v > 1e-5
               for v in turn['rollout_logprobs']):
            raise ParityError('Invalid raw action probability')
        if not ID.fullmatch(turn['request_id']) or any(t['request_id'] == turn['request_id'] for t in self._turns):
            raise ParityError('Repeated/invalid engine request')
        if self._turns and not can_extend(self._turns[-1], turn):
            raise ParityError('Append-only input/media prefix was rewritten')
        self._turns.append(turn)
        self._ids.extend(turn['response_ids'])
        self._pending = None
        self._phase = 'observing'

    def append(self, segment):
        if self._phase != 'observing' or self.reason:
            raise ParityError('Observation outside a completed policy turn')
        segment = checked_segment(segment)
        # Full observation was journaled before this call. If it cannot fit,
        # retain delivery/actions and end, never rewrite/remove old context.
        if (len(self._ids) + len(segment['token_ids']) >= self.limits.context or
                len(self._features) + len(segment['features']) > MAX_CONTEXT_IMAGES):
            self.finish('context_limit')
            return False
        base = len(self._ids)
        self._ids.extend(segment['token_ids'])
        self._features.extend({**f, 'offset': base+f['offset']} for f in segment['features'])
        self._phase = 'ready'
        return True

    def finish(self, reason):
        if self.reason or self._phase == 'generating':
            raise ParityError('Cannot mark an in-flight/finished request successful')
        if reason not in ('final', 'context_limit', 'turn_limit', 'output_limit') or not self._turns:
            raise ParityError('Invalid successful trajectory termination')
        self.reason, self._phase = reason, 'finished'

    def packet(self):
        if self._phase != 'finished':
            raise ParityError('Unfinished/failed trajectory is not training data')
        packets = plan_chains(self._turns)
        if len(packets) != 1:
            raise ParityError('Append-only trajectory split into multiple training packets')
        return packets[0]


async def rollout(*, initial, engine, renderer, tools, journal, attempt_id, session_id,
                  contract_sha256, limits=Limits(), turn_observer=None, budget=None, budget_observer=None):
    """Sole entry for train and validation; no caller-selectable loop variants.

    The journal is trusted/durable. The engine returns {turn, action}, where
    `turn` is independently sealed/captured and `action` is parsed only for tool
    dispatch. Tool/action text is never used to reconstruct training targets.
    """
    state = AppendState(initial, attempt_id=attempt_id, session_id=session_id,
                        contract_sha256=contract_sha256, limits=limits)
    await journal.event('loop_start', {'identity': state.identity, 'limits': limits.__dict__, 'initial': initial})
    try:
        while not state.reason:
            request = state.request()
            await journal.event('generation_requested', request)
            result = await engine.generate(deepcopy(request))
            state.accept(result['turn'])
            action = result['action']
            await journal.event('generation_complete', deepcopy(result))
            async def observe(observation=None):
                if turn_observer is not None:
                    await turn_observer(turn=state.turn_count, request_id=result['turn']['request_id'],
                        action=deepcopy(action), observation=deepcopy(observation))
            if result['turn']['finish_reason'] == 'length':
                await observe()
                state.finish('context_limit' if len(request['prompt_token_ids']) +
                             len(result['turn']['response_ids']) >= limits.context else 'output_limit')
                break
            if type(action) is not dict or set(action) != {'text', 'calls'} or type(action['calls']) is not list:
                raise ParityError('Malformed parser result from generation backend')
            if not action['calls']:
                await observe()
                state.finish('final')
                break
            await journal.event('tool_step_started', action)
            observation = await tools.step(deepcopy(action))
            await journal.event('tool_step_complete', deepcopy(observation))
            await observe(observation)
            if state.turn_count >= limits.turns:
                state.finish('turn_limit')
                break
            # Only NEW observations cross this boundary. No prior assistant
            # text, reasoning field or conversation list is given to renderer.
            notice = budget.after_tool(completed_turn=state.turn_count,
                tokens_used=len(request['prompt_token_ids']) + len(result['turn']['response_ids'])) if budget is not None else None
            render_kwargs = {'last_token_id': result['turn']['response_ids'][-1],
                'final_turn': state.turn_count + 1 == limits.turns}
            if notice is not None:
                render_kwargs['budget_notice'] = notice['text']
            suffix = await renderer.observation(deepcopy(observation), **render_kwargs)
            await journal.event('observation_rendered', deepcopy(suffix))
            appended = state.append(suffix)
            if notice is not None:
                await journal.event('harness_budget_notice', {**notice, 'appended_to_context': appended})
                if appended and budget_observer is not None:
                    budget_observer(deepcopy(notice))
        packet = state.packet()
        output = {'reason': state.reason, 'request_ids': state.request_ids,
                  'packet': packet, 'contract_sha256': contract_sha256}
        await journal.event('loop_complete', output)
        await journal.seal('complete')
        return output
    except BaseException as exc:
        await journal.event('loop_failed', {'type': type(exc).__name__, 'error': str(exc)[:2000],
                                           'completed_request_ids': state.request_ids})
        await journal.seal('failed')
        raise
