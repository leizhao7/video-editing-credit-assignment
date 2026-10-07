"""Trusted raw-token backend. Native vLLM renders/parses, hooks prove admission.

Only the parent controller can reach this service (localhost). Policy tools
have no inference HTTP endpoint or credentials. Image tensors are persisted
once and referenced on subsequent turns, not resent over the parent channel.
"""
import asyncio
from .copying import deepcopy
import json
from types import SimpleNamespace

from opencode_rl.capacity import durable_io
from opencode_rl.contract import ID, ParityError, canonical, digest
from opencode_rl.records import EngineCapture
from opencode_rl.store import atomic_bytes
from opencode_rl.vllm_hooks import ACTIVE, feature_records
from opencode_rl.visual_budget import MAX_CONTEXT_IMAGES
from .loop import checked_segment


def metadata(value, contract):
    keys = {'attempt_id', 'session_id', 'contract_sha256', 'agent', 'primary', 'turn'}
    if (type(value) is not dict or set(value) != keys or
            any(not isinstance(value[k], str) or not ID.fullmatch(value[k]) for k in ('attempt_id', 'session_id')) or
            value['contract_sha256'] != contract.sha256 or value['agent'] != 'build' or value['primary'] is not True or
            type(value['turn']) is not int or not 0 <= value['turn'] < contract.max_turns):
        raise ParityError('Invalid append controller provenance')
    return {**deepcopy(value), 'logical_call_id': f"append-turn-{value['turn']}"}


class CheckedCapture(EngineCapture):
    """Compare actual post-processor input before EngineCore is called."""
    def __init__(self, journal, contract, version, expected):
        super().__init__(journal, contract, version)
        self.expected = deepcopy(expected)
        self.finish_reason = None

    def start(self, prompt_ids, features, sampling, **kwargs):
        if prompt_ids != self.expected['prompt_token_ids'] or features != self.expected['features']:
            raise ParityError('Engine admission rewrote append-only tokens/image features')
        if sampling.get('max_tokens') != self.expected['max_tokens']:
            raise ParityError('Engine admission changed the remaining output budget')
        super().start(prompt_ids, features, sampling, **kwargs)

    def output(self, *args, **kwargs):
        super().output(*args, **kwargs)
        if self.finished:
            self.finish_reason = kwargs['finish_reason']

    def turn(self, meta):
        if not self.journal.sealed or not self.finished or not self.ids or self.finish_reason not in ('stop', 'length'):
            raise ParityError('Unsealed/incomplete raw-token generation')
        return {'request_id': self.journal.request_id, 'metadata': deepcopy(meta), 'input': deepcopy(self.input),
                'response_ids': list(self.ids), 'rollout_logprobs': list(self.logprobs),
                'loss_mask': [1]*len(self.ids), 'finish_reason': self.finish_reason}


class FeatureRegistry:
    def __init__(self, store, contract):
        self.store, self.contract = store, contract

    def path(self, identifier):
        return self.store.root / 'append_mm_transport' / (digest({
            'processor': self.contract.processor_sha256, 'identifier': identifier}) + '.json')

    def register(self, engine_input):
        if engine_input.get('type') != 'multimodal':
            return []
        from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import encode_mm_kwargs_item
        hashes, positions, items = (engine_input[k] for k in ('mm_hashes', 'mm_placeholders', 'mm_kwargs'))
        if set(hashes) != {'image'} or set(positions) != {'image'} or set(items) != {'image'}:
            raise ParityError('Only Qwen image tensors are supported')
        if not len(hashes['image']) == len(positions['image']) == len(items['image']) <= MAX_CONTEXT_IMAGES:
            raise ParityError('Image transport inventory mismatch')
        features = []
        for identifier, position, item in zip(hashes['image'], positions['image'], items['image']):
            # The upstream scale-out wire format loses sparse masks. Reject
            # them explicitly rather than silently changing MM positions.
            if getattr(position, 'is_embed', None) is not None or item is None:
                raise ParityError('Sparse/cache-only rendering needs separate parity support')
            feature = feature_records(self.store, self.contract, [SimpleNamespace(
                modality='image', identifier=identifier, mm_position=position, data=item)])[0]
            path = self.path(identifier)
            with self.store.mm_index_lock:
                if path.exists():
                    old = json.loads(path.read_bytes())
                    if old['data'] != feature['data']:
                        raise ParityError('Image transport identity collision')
                else:
                    blob = self.store.blob(encode_mm_kwargs_item(item).encode('ascii'))
                    atomic_bytes(path, canonical({'identifier': identifier, 'data': feature['data'], 'encoded': blob}))
            features.append(feature)
        return features

    def transport(self, features):
        if not features:
            return None
        if len(features) > MAX_CONTEXT_IMAGES:
            raise ParityError('Image context exceeds the pinned 999-image limit')
        hashes, positions, data = [], [], []
        for feature in features:
            record = json.loads(self.path(feature['identifier']).read_bytes())
            if record['identifier'] != feature['identifier'] or record['data'] != feature['data']:
                raise ParityError('Image reference differs from the original renderer tensors')
            hashes.append(feature['identifier'])
            positions.append({'offset': feature['offset'], 'length': feature['length']})
            data.append(self.store.read_blob(record['encoded']).decode('ascii'))
        return {'mm_hashes': {'image': hashes}, 'mm_placeholders': {'image': positions},
                'kwargs_data': {'image': data}}


def parsed_action(message, turn):
    """Dispatch-only view. Invalid model arguments remain an observed action."""
    calls = []
    for index, call in enumerate(message.get('tool_calls') or []):
        function = call.get('function', {})
        name = function.get('name', '')
        try:
            arguments = json.loads(function['arguments'])
            if type(arguments) is not dict or not isinstance(name, str) or not name:
                raise ValueError('Tool arguments must be an object and name must be present')
        except (KeyError, TypeError, ValueError) as exc:
            arguments = {'tool': str(name)[:160], 'error': str(exc)[:1000]}
            name = 'invalid'
        calls.append({'id': f't{turn}-c{index}', 'name': name, 'arguments': arguments})
    if len(calls) > 64:
        # The whole raw response still trains. Do not execute a truncated
        # prefix of a huge model-generated call batch.
        calls = [{'id': f't{turn}-overflow', 'name': 'invalid',
                  'arguments': {'tool': 'batch', 'error': 'At most 64 calls are allowed per turn'}}]
    return {'text': canonical(message).decode(), 'calls': calls}


async def drain_native_tokens(stream, capture):
    """Consume native DELTA SSE in-process; no quadratic cumulative LP copies.

    Raw LPs and partial actions are already durably captured by engine hooks.
    This independently checks the native token transport's termination/IDs.
    """
    ids, finish, done = [], None, False
    try:
        async for chunk in stream:
            if done:
                raise ParityError('Native token stream continued after DONE')
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            for line in text.splitlines():
                if not line.strip():
                    continue
                if not line.startswith('data: '):
                    raise ParityError('Unreviewed native token stream framing')
                payload = line[6:]
                if payload == '[DONE]':
                    done = True
                    continue
                if done:
                    raise ParityError('Native token stream continued after DONE')
                value = json.loads(payload)
                choices = value.get('choices')
                if type(choices) is not list or len(choices) > 1 or value.get('error'):
                    raise ParityError('Native token stream returned an error/fanout')
                for choice in choices:
                    delta = choice.get('token_ids') or []
                    if (choice.get('index') != 0 or type(delta) is not list or
                            any(type(t) is not int or t < 0 for t in delta) or finish is not None):
                        raise ParityError('Native token delta is invalid or follows termination')
                    ids.extend(delta)
                    if choice.get('finish_reason') is not None:
                        finish = choice['finish_reason']
        # Pinned ServingTokens skips empty token deltas, including an empty
        # terminal delta. EngineCapture still observes that termination. Only
        # its actual finished event may supply a missing wire finish_reason.
        if (not done or not ids or not capture.finished or ids != capture.ids or
                capture.finish_reason not in ('stop', 'length') or
                finish not in (None, capture.finish_reason)):
            raise ParityError('Native token stream is incomplete')
        return ids, capture.finish_reason
    finally:
        await stream.aclose()


class Backend:
    def __init__(self, state, store, contract, policy, *, model='qwen35-9b-policy', render_concurrency=16):
        if type(render_concurrency) is not int or not 1 <= render_concurrency <= 64:
            raise ParityError('Rendering must have a bounded CPU admission gate')
        self.state, self.store, self.contract, self.policy, self.model = state, store, contract, policy, model
        self.features = FeatureRegistry(store, contract)
        self.render_gate = asyncio.Semaphore(render_concurrency)

    def chat_request(self, messages, tools, add_generation_prompt=True):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        return ChatCompletionRequest(model=self.model, messages=messages, tools=tools or None,
            tool_choice='auto' if tools else 'none', stream=False, temperature=1.0, top_p=1.0, top_k=-1,
            max_tokens=1, add_generation_prompt=add_generation_prompt,
            chat_template_kwargs={'enable_thinking': self.contract.thinking})

    async def render(self, *, messages, tools, add_generation_prompt):
        # No weight lease is held during independent tokenizer/vision CPU work.
        image_count = sum(part.get('type') == 'image_url' for message in messages
                          if isinstance(message.get('content'), list)
                          for part in message['content'] if isinstance(part, dict))
        request = self.chat_request(messages, tools, add_generation_prompt)
        async with self.render_gate:
            result = await self.state.online_renderer.render_chat(request, skip_mm_cache=True)
            if not isinstance(result, tuple) or len(result[1]) != 1:
                raise ParityError('Native vLLM renderer did not return one complete input')
            engine_input = result[1][0]
            if engine_input.get('type') not in ('token', 'multimodal'):
                raise ParityError('Only actual token IDs and image tensors may be rendered')
            features = await durable_io(self.features.register, engine_input)
            if len(features) != image_count:
                raise ParityError('Native renderer changed the number of tool/prompt images')
        return checked_segment({'token_ids': list(engine_input['prompt_token_ids']), 'features': features})

    def markers(self):
        tokenizer = self.state.online_renderer.renderer.get_tokenizer()
        end = tokenizer.encode('<|im_end|>', add_special_tokens=False)
        if len(end) != 1 or tokenizer.decode(end, skip_special_tokens=False) != '<|im_end|>':
            raise ParityError('The pinned Qwen end marker changed')
        return {'assistant_end_ids': end, 'newline_ids': tokenizer.encode('\n', add_special_tokens=False)}

    async def generate(self, request, *, tools):
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            GenerateRequest, GenerateResponse, GenerateResponseChoice, DerenderChatRequest)
        from vllm.sampling_params import SamplingParams, RequestOutputKind
        meta = metadata(request['metadata'], self.contract)
        segment = checked_segment({'token_ids': request['prompt_token_ids'], 'features': request['features']})
        remaining = self.contract.max_context - len(segment['token_ids'])
        if type(request['max_tokens']) is not int or request['max_tokens'] != min(self.contract.max_output, remaining) or remaining <= 0:
            raise ParityError('Incorrect remaining raw-token generation budget')
        journal = await durable_io(self.store.new_request, meta, {'format': 'append-token-input-v1', **request})
        token = None
        try:
            # Keep the new cancellation point inside journal cleanup.
            # durable_io drains accepted copying before sealing failure.
            capture = await durable_io(CheckedCapture, journal, self.contract, self.policy.version, request)
            features = await durable_io(self.features.transport, segment['features'])
            native = GenerateRequest(request_id=journal.request_id, model=self.model, stream=True,
                token_ids=segment['token_ids'], features=features,
                sampling_params=SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1,
                    # The parent HTTP response is nonstream; native generation
                    # remains DELTA so partial failed actions are persisted and
                    # old IDs/LPs are not recopied at every generated token.
                    max_tokens=request['max_tokens'], logprobs=0, output_kind=RequestOutputKind.DELTA))
            token = ACTIVE.set(capture)
            stream = await self.state.serving_tokens.serve_tokens(native)
            if not hasattr(stream, '__aiter__'):
                raise ParityError('Native raw-token generation did not return a stream')
            ids, finish = await drain_native_tokens(stream, capture)
            if not capture.finished or ids != capture.ids or finish != capture.finish_reason:
                raise ParityError('Raw-token response differs from independently captured output')
            response = GenerateResponse(request_id=capture.input['engine_request_id'], model=self.model,
                choices=[GenerateResponseChoice(index=0, token_ids=ids, finish_reason=finish)])
            # Parser is the same pinned native reasoning/tool parser. It only
            # controls dispatch; no decoded/re-encoded assistant enters targets.
            parsed = await self.state.serving_derender.derender_chat_response(DerenderChatRequest(
                model=self.model, generate_response=response, prompt_tokens=len(segment['token_ids']),
                chat_request=self.chat_request([{'role': 'user', 'content': ''}], tools)))
            if not hasattr(parsed, 'choices') or len(parsed.choices) != 1:
                raise ParityError('Native Qwen output parser failed')
            message = parsed.choices[0].message.model_dump(mode='json')
            action = parsed_action(message, meta['turn'])
            await durable_io(journal.event, 'parsed_action', {'message': message, 'action': action})
            await durable_io(journal.event, 'wire_complete', {'done': True, 'transport': 'native-token-delta-stream'})
            await durable_io(journal.seal, 'complete')
            return {'turn': await durable_io(capture.turn, meta), 'action': action}
        except BaseException as exc:
            if not journal.sealed:
                await durable_io(journal.event, 'error', {'type': type(exc).__name__, 'message': str(exc)[:2000]})
                await durable_io(journal.seal, 'failed')
            raise
        finally:
            if token is not None:
                ACTIVE.reset(token)
