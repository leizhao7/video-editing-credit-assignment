"""Observe vLLM 0.26.0 at EngineCore admission and raw generation output.

This is installed ONLY in the isolated OpenCode inference server. The official
chat parser and SSE stream stay intact; no completion text is re-tokenized.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import json
import time
from pathlib import Path

from .contract import ID, ParityError, canonical, digest
from .records import EngineCapture
from .store import atomic_bytes
from .capacity import durable_io

SOURCE_PINS = {
    "config/scheduler.py": "a816cf79a3e74ffc0984f9bebb274275b26f46be8b28cb77a29388a0996263c8",
    "engine/arg_utils.py": "e46d992fb2553f7ecd71d9b64459ecbc7daa740305b19e862493389d1b68d11a",
    "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
    "v1/engine/output_processor.py": "ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b",
    "v1/engine/async_llm.py": "d49ae7978db5653a6f4fb897298b144a6044931f7765e7b34ea5f6c0f2da6b87",
    "entrypoints/openai/chat_completion/serving.py": "860e27d548ec7bb0497a46e0356a5acc5500998944cffa5e67e336041693c067",
    "v1/engine/input_processor.py": "c5673988c0f7cfec268220e3f044e718702c015a4f236c020937cfd40a793f15",
    "multimodal/inputs.py": "18f8549124fab2c0012b3a1f49aeec10d4abc8423d66378124049ea776092c37",
    "multimodal/cache.py": "f44e46a1f608578f3f73c9fc260de04f3d70bc8024a4d6e03e47277913fc7205",
    "multimodal/registry.py": "d4627c89d3459ce35a729e0940a0c5be5b54793b40192c1c068416b57a0a82fd",
    "renderers/base.py": "d3f748aafa3729fa0a27f29ed7e845e60460148a18c11c893ba02c638e9fd644",
}
ACTIVE = contextvars.ContextVar("opencode_rl_capture", default=None)


def verify_vllm_source(root):
    for relative, sha in SOURCE_PINS.items():
        if hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest() != sha:
            raise ParityError(f"Unreviewed vLLM source: {relative}")


def feature_records(store, contract, features):
    result = []
    previous_end = -1
    for feature in features or []:
        if feature.modality != "image":
            raise ParityError("Only image tensors are currently wired to Qwen training")
        position = feature.mm_position
        if position.offset < previous_end or position.length <= 0:
            raise ParityError("Invalid/overlapping image token positions")
        previous_end = position.offset + position.length
        identity = digest({"processor": contract.processor_sha256, "identifier": feature.identifier})
        cache = store.root / "mm_input_index" / f"{identity}.json"
        if feature.data is None:
            if not cache.exists():
                raise ParityError("Multimodal cache hit has no durable input provenance")
            data = json.loads(cache.read_bytes())
            for ref in data.values():
                store.load_tensor(ref)
        else:
            raw = feature.data.get_data()
            if set(raw) != {"pixel_values", "image_grid_thw"}:
                raise ParityError(f"Unreviewed Qwen image tensor fields: {sorted(raw)}")
            data = {key: store.tensor(value) for key, value in raw.items()}
            # One TraceStore per dedicated server. Compare and publish together
            # so concurrent writers cannot silently overwrite a conflicting ID.
            with store.mm_index_lock:
                if cache.exists():
                    if json.loads(cache.read_bytes()) != data:
                        raise ParityError("Multimodal identifier resolved to different input tensors")
                else:
                    atomic_bytes(cache, canonical(data))
        result.append({"modality": "image", "identifier": feature.identifier,
                       "offset": position.offset, "length": position.length, "data": data})
    return result


class StaticPolicy:
    """Fixed-weight diagnostic server; deliberately NOT an async reload adapter."""
    def __init__(self, version):
        if not version:
            raise ParityError("A real checkpoint/version identity is required")
        self.version = version

    async def acquire(self, tokens=0):
        return self.version

    async def release(self, tokens=0):
        pass


def request_metadata(headers, contract):
    result = {key: headers.get(f"x-ocrl-{name}", "") for key, name in (
        ("session_id", "session"), ("logical_call_id", "call"),
        ("agent", "agent"), ("attempt_id", "attempt"),
    )}
    if any(not ID.fullmatch(value) for value in result.values()):
        raise ParityError("Missing/invalid OpenCode provenance headers")
    if headers.get("x-ocrl-contract") != contract.sha256:
        raise ParityError("OpenCode runtime contract mismatch")
    result["contract_sha256"] = contract.sha256
    result["primary"] = headers.get("x-ocrl-primary") == "1"
    try:
        result["turn"] = int(headers["x-ocrl-turn"])
    except (KeyError, ValueError) as exc:
        raise ParityError("Missing turn provenance") from exc
    if result["turn"] < 0 or (result["primary"] and result["agent"] != "build"):
        raise ParityError("Invalid main-agent identity")
    return result


def install(store, contract, policy, *, feature_preparation_concurrency=8):
    """Return restoration callback. Call before the dedicated server starts."""
    if type(feature_preparation_concurrency) is not int or feature_preparation_concurrency < 1:
        raise ValueError('Positive feature preparation concurrency required')
    # Bound preparation plus the ready-to-admit queue. Release this slot as
    # soon as native admission returns, NOT when GPU generation finishes.
    feature_gate = asyncio.Semaphore(feature_preparation_concurrency)
    import msgspec
    import vllm
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.sampling_params import RequestOutputKind
    from vllm.multimodal.registry import MultiModalRegistry
    from vllm.renderers.base import BaseRenderer
    from .multimodal_cache import install_cache_hooks

    verify_vllm_source(Path(vllm.__file__).parent)
    if getattr(AsyncLLM, "_opencode_rl_installed", False):
        raise ParityError("Capture hooks already installed")
    original_chat = OpenAIServingChat.create_chat_completion
    original_add = AsyncLLM._add_request
    original_generate = AsyncLLM.generate
    restore_cache = install_cache_hooks(MultiModalRegistry, BaseRenderer)

    async def admitted(engine, request, prompt, parent_req, index, queue):
        capture = ACTIVE.get()
        if capture is None or parent_req is not None or index != 0:
            raise ParityError("Untraced/fan-out engine admission rejected")
        if request.prompt_embeds is not None or request.prompt_is_token_ids is not None or request.lora_request is not None:
            raise ParityError("Embeddings/LoRA require separate parity support")
        # Admission uses EXACT post-tokenization/image-expansion length, not a
        # text-byte estimate. Reserve possible output too. CPU rendering is
        # weight-independent; acquire before adding anything to EngineCore.
        tokens = min(contract.max_context, len(request.prompt_token_ids or []) + request.sampling_params.max_tokens)
        preparation_queued = time.monotonic()
        await durable_io(capture.journal.event, 'engine_feature_queue_enter', {'time': time.time()})
        async with feature_gate:
            await durable_io(capture.journal.event, 'engine_feature_queue_exit',
                            {'wait_seconds': time.monotonic() - preparation_queued, 'time': time.time()})
            feature_started = time.monotonic()
            # Raw processor pixels/grids do not depend on policy weights.
            # Persist them BEFORE reserving tokens or holding a weight lease.
            # Self-contained MM cache hooks make reordered admission safe.
            features = await durable_io(feature_records, store, contract, request.mm_features)
            await durable_io(capture.journal.event, 'engine_features_saved',
                             {'seconds': time.monotonic() - feature_started, 'images': len(features),
                              'time': time.time()})
            sampling = {key: msgspec.to_builtins(getattr(request.sampling_params, key))
                        for key in request.sampling_params.__struct_fields__}
            mode = engine.model_config.logprobs_mode
            queued = time.monotonic()
            await durable_io(capture.journal.event, 'engine_queue_enter', {'reserved_tokens': tokens})
            version = await policy.acquire(tokens)
            capture.lease_tokens = tokens
            capture.weight_version = version
            await durable_io(capture.journal.event, 'engine_queue_exit',
                            {'reserved_tokens': tokens, 'wait_seconds': time.monotonic() - queued,
                             'weight_version': version, 'time': time.time()})
            # The actual acquired version, not the version at HTTP/preparation
            # start, binds the input. Capture still precedes native execution.
            await durable_io(capture.start, list(request.prompt_token_ids or []), features, sampling,
                             engine_request_id=request.request_id, logprobs_mode=mode)
            return await original_add(engine, request, prompt, parent_req, index, queue)

    async def generate(engine, prompt, sampling_params, request_id, **kwargs):
        capture = ACTIVE.get()
        if capture is None:
            raise ParityError("Generation without an actual OpenCode request")
        from vllm.v1.engine import EngineCoreRequest
        if isinstance(prompt, EngineCoreRequest):
            raise ParityError("Prebuilt engine requests bypass sampling metadata overrides")
        params = copy.copy(sampling_params)
        params.logprobs = 0  # Metadata only; do not alter sampling or stop semantics.
        capture.lease_tokens = None
        generation = original_generate(engine, prompt, params, request_id, **kwargs)
        try:
            async for output in generation:
                if len(output.outputs) != 1:
                    raise ParityError("Expected a single raw engine completion")
                completion = output.outputs[0]
                ids = list(completion.token_ids)
                if completion.logprobs is None or len(completion.logprobs) != len(ids):
                    raise ParityError("Raw sampled probabilities missing")
                lps = []
                for tid, options in zip(ids, completion.logprobs):
                    value = options.get(tid) if options else None
                    if value is None:
                        raise ParityError("Sampled token has no actual behavior logprob")
                    lps.append(value.logprob)
                await durable_io(capture.output, ids, lps,
                    cumulative=params.output_kind != RequestOutputKind.DELTA,
                    finished=output.finished, finish_reason=completion.finish_reason,
                    weight_version=policy.version)
                yield output
        finally:
            # Disconnect/cancellation must not strand a reader and deadlock the
            # next weight transaction. Close/abort the native generator BEFORE
            # releasing its lease, including errors raised by our own recorder.
            async def cleanup():
                try:
                    await generation.aclose()
                except BaseException:
                    if hasattr(policy, 'fail'):
                        await policy.fail()
                    raise
                finally:
                    if capture.lease_tokens is not None:
                        await policy.release(capture.lease_tokens)
                        capture.lease_tokens = None
            release = asyncio.create_task(cleanup())
            try:
                await asyncio.shield(release)
            except asyncio.CancelledError:
                await release
                raise

    async def chat(serving, request, raw_request=None):
        if raw_request is None or not request.stream:
            raise ParityError("Dedicated OpenCode server requires a real streaming HTTP request")
        if serving._effective_chat_template_kwargs(request).get("enable_thinking") is not contract.thinking:
            raise ParityError("Actual serving template thinking mode differs from contract")
        meta = request_metadata(raw_request.headers, contract)
        body = await raw_request.json()
        journal = await durable_io(store.new_request, meta, body)
        capture = EngineCapture(journal, contract, policy.version)
        token = ACTIVE.set(capture)
        try:
            result = await original_chat(serving, request, raw_request)
        except BaseException as exc:
            await durable_io(journal.event, "error", {"type": type(exc).__name__, "message": str(exc)})
            from .context_recovery import is_context_message
            if capture.input is None and type(exc).__name__ == 'VLLMValidationError' and is_context_message(str(exc)):
                await durable_io(journal.event, 'context_overflow_rejected', {'before_engine_admission': True, 'generated_tokens': 0})
                await durable_io(journal.seal, 'context_overflow')
            else:
                await durable_io(journal.seal, "failed")
            raise
        finally:
            ACTIVE.reset(token)
        if not hasattr(result, "__aiter__"):
            await durable_io(journal.event, "error", {"type": "non_stream_response", "message": str(result)})
            await durable_io(journal.seal, "failed")
            return result

        async def stream():
            token = ACTIVE.set(capture)
            done = False
            try:
                async for chunk in result:
                    body = chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                    blob = await durable_io(store.blob, body)
                    await durable_io(journal.event, "wire_chunk", {"blob": blob})
                    done |= any(line.strip() == b"data: [DONE]" for line in body.splitlines())
                    yield chunk
                if not done or not capture.finished:
                    raise ParityError("HTTP stream or engine output did not complete")
                await durable_io(journal.event, "wire_complete", {"done": True})
                await durable_io(journal.seal, "complete")
            except BaseException as exc:
                if not journal.sealed:
                    await durable_io(journal.event, "error", {"type": type(exc).__name__, "message": str(exc)})
                    await durable_io(journal.seal, "failed")
                raise
            finally:
                ACTIVE.reset(token)
                await result.aclose()
        return stream()

    AsyncLLM._add_request = admitted
    AsyncLLM.generate = generate
    OpenAIServingChat.create_chat_completion = chat
    AsyncLLM._opencode_rl_installed = True

    def restore():
        AsyncLLM._add_request = original_add
        AsyncLLM.generate = original_generate
        OpenAIServingChat.create_chat_completion = original_chat
        restore_cache()
        del AsyncLLM._opencode_rl_installed
    return restore
