"""Per-engine read leases and a cluster-wide, fail-closed weight transaction.

The transaction wraps SkyRL's *entire* save_weights_for_sampler, before its
KEEP-pause. Draining after pause would deadlock the very readers being drained.
Only the independent OpenCode image installs this decorator and the HTTP routes.
"""
from __future__ import annotations

import asyncio
import functools
import time
import uuid
from collections import deque

from .contract import ID, ParityError, canonical
from .store import atomic_bytes


class AsyncPolicy:
    def __init__(self, version, *, audit_root=None, max_readers=24, token_budget=786432, queue_timeout=540):
        if not version:
            raise ParityError('Missing initial checkpoint identity')
        self.version = version
        self.phase, self.transaction, self.next_version = 'open', None, None
        self.readers = 0
        if max_readers < 1 or token_budget < 1 or queue_timeout <= 0:
            raise ParityError('Invalid inference admission capacity')
        self.max_readers, self.token_budget, self.queue_timeout = max_readers, token_budget, queue_timeout
        self.reserved_tokens = 0
        self.waiters = deque()
        self.admitted = self.queue_timeouts = 0
        self.wait_seconds_sum = self.wait_seconds_max = 0.
        self.condition = asyncio.Condition()
        self.audit_root = audit_root

    def status(self):
        return {'version': self.version, 'phase': self.phase, 'readers': self.readers,
                'transaction': self.transaction, 'next_version': self.next_version,
                'waiting': len(self.waiters), 'reserved_tokens': self.reserved_tokens,
                'max_readers': self.max_readers, 'token_budget': self.token_budget,
                'admitted': self.admitted, 'queue_timeouts': self.queue_timeouts,
                'wait_seconds_sum': self.wait_seconds_sum, 'wait_seconds_max': self.wait_seconds_max}

    async def _audit(self, event):
        if self.audit_root is not None:
            await asyncio.to_thread(atomic_bytes, self.audit_root / f'{uuid.uuid4().hex}.json',
                                    canonical({'event': event, **self.status()}))

    async def acquire(self, tokens=0):
        if type(tokens) is not int or not 0 <= tokens <= self.token_budget:
            raise ParityError('Request exceeds engine token reservation capacity; never truncate it')
        ticket, start = object(), time.monotonic()
        async with self.condition:
            self.waiters.append(ticket)
            try:
                # FIFO bounds starvation. Waiting requests own NO read lease,
                # so a weight drain waits only for requests actually admitted.
                async with asyncio.timeout(self.queue_timeout):
                    await self.condition.wait_for(lambda: self.phase == 'failed' or (
                        self.phase == 'open' and self.waiters[0] is ticket and
                        self.readers < self.max_readers and self.reserved_tokens + tokens <= self.token_budget))
                if self.phase == 'failed':
                    raise ParityError('Weight transaction failed; this engine cannot generate')
                self.readers += 1
                self.reserved_tokens += tokens
                waited = time.monotonic() - start
                self.admitted += 1
                self.wait_seconds_sum += waited
                self.wait_seconds_max = max(self.wait_seconds_max, waited)
                return self.version
            except TimeoutError:
                self.queue_timeouts += 1
                raise
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    async def release(self, tokens=0):
        async with self.condition:
            if self.readers <= 0 or type(tokens) is not int or not 0 <= tokens <= self.reserved_tokens:
                raise ParityError('Unbalanced generation lease')
            self.readers -= 1
            self.reserved_tokens -= tokens
            self.condition.notify_all()

    async def drain(self, transaction, expected_version, timeout=1200):
        if not ID.fullmatch(transaction):
            raise ParityError('Invalid weight transaction ID')
        try:
            async with self.condition:
                if self.phase != 'open' or self.version != expected_version:
                    raise ParityError('Weight drain state/version conflict')
                self.phase, self.transaction = 'draining', transaction
                await self._audit('drain_begin')
                async with asyncio.timeout(timeout):
                    await self.condition.wait_for(lambda: self.readers == 0 or self.phase == 'failed')
                if self.phase == 'failed':
                    raise ParityError('Drain was aborted')
                self.phase = 'updating'
                await self._audit('drained')
                return self.status()
        except BaseException:
            await self.fail()
            raise

    async def prepare(self, transaction, version, reset_cache):
        try:
            async with self.condition:
                if self.phase != 'updating' or self.readers or transaction != self.transaction or version == self.version:
                    raise ParityError('Invalid weight commit preparation')
                # This is checked directly, not SkyRL's endpoint that discards
                # vLLM's boolean reset result. No requests are resident now.
                result = await reset_cache()
                if result is not True:
                    raise ParityError(f'KV cache invalidation was not acknowledged: {result!r}')
                self.next_version, self.phase = version, 'prepared'
                await self._audit('cache_invalidated')
                return self.status()
        except BaseException:
            await self.fail()
            raise

    async def open(self, transaction, version):
        async with self.condition:
            if self.phase != 'prepared' or self.transaction != transaction or self.next_version != version:
                raise ParityError('Invalid weight commit')
            self.version, self.phase, self.next_version = version, 'open', None
            await self._audit('committed')
            self.condition.notify_all()
            return self.status()

    async def fail(self):
        async with self.condition:
            self.phase = 'failed'
            self.condition.notify_all()
            await self._audit('failed')


async def cluster_call(client, endpoint, body, *, phase=None, version=None):
    replies = await client._call_all_servers(endpoint, json=body)
    if set(replies) != set(client.server_urls):
        raise ParityError('Missing inference engine acknowledgement')
    for reply in replies.values():
        state = reply.get('body') or {}
        if reply.get('status') != 200 or (phase and state.get('phase') != phase) or (version and state.get('version') != version):
            raise ParityError(f'Invalid inference engine acknowledgement: {state}')
    return replies


def weight_transaction(function):
    @functools.wraps(function)
    async def wrapped(dispatch, *args, **kwargs):
        if dispatch.colocate_all or dispatch.cfg.trainer.policy.model.lora.rank:
            raise ParityError('OpenCode weight transactions require separate full-parameter engines')
        client = dispatch._inference_engine_client
        if not getattr(client, 'server_urls', None):
            raise ParityError('OpenCode async requires the reviewed native HTTP engines')
        token = uuid.uuid4().hex
        from .contract import MODEL_REVISION
        before = f'{MODEL_REVISION}:{client.weight_version}'
        try:
            async with asyncio.timeout(1800):
                await cluster_call(client, '/ocrl/drain', {'transaction': token, 'expected_version': before},
                                   phase='updating', version=before)
                result = await function(dispatch, *args, **kwargs)
                after = f'{MODEL_REVISION}:{client.weight_version}'
                if after == before:
                    raise ParityError('SkyRL did not advance its weight version')
                # All transfers and resets must succeed before ANY gate opens.
                await cluster_call(client, '/ocrl/prepare', {'transaction': token, 'version': after}, phase='prepared')
                await cluster_call(client, '/ocrl/open', {'transaction': token, 'version': after}, phase='open', version=after)
                return result
        except BaseException:
            # Even if upstream finally resumed its scheduler, these gates remain
            # closed. Do not retry an ambiguous/partial NCCL update in place.
            task = asyncio.create_task(client._call_all_servers('/ocrl/fail', json={}))
            try:
                await asyncio.wait_for(asyncio.shield(task), 15)
            except BaseException:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise
    return wrapped


def add_routes(app, engine, policy):
    from fastapi.responses import JSONResponse

    @app.post('/ocrl/drain')
    async def drain(body: dict):
        return await policy.drain(body['transaction'], body['expected_version'])

    @app.post('/ocrl/prepare')
    async def prepare(body: dict):
        async def reset_all_weight_dependent_caches():
            result = await engine.reset_prefix_cache(reset_running_requests=True)
            if result is not True:
                return result
            # Image embeddings depend on vision weights too. Raw processor
            # pixel caches may survive, but encoded image features must not.
            await engine.reset_encoder_cache()
            return True
        return await policy.prepare(body['transaction'], body['version'], reset_all_weight_dependent_caches)

    @app.post('/ocrl/open')
    async def open_gate(body: dict):
        return await policy.open(body['transaction'], body['version'])

    @app.post('/ocrl/fail')
    async def fail(body: dict):
        await policy.fail()
        return policy.status()

    @app.get('/ocrl/status')
    async def status():
        return policy.status()

    async def prevent_unguarded_update(request, call_next):
        path = request.url.path
        mutation = path in ('/update_weights', '/start_weight_update', '/finish_weight_update',
                            '/fetch_weights', '/reset_prefix_cache', '/load_lora_adapter')
        if path == '/collective_rpc':
            body = await request.json()
            # Only these initialization/read methods are valid outside a lease.
            mutation = body.get('method') not in ('get_world_size', 'init_weight_transfer', 'init_weight_update_group')
        if mutation and policy.phase != 'updating':
            return JSONResponse({'error': 'Weight mutation outside a drained transaction'}, status_code=409)
        return await call_next(request)
    # vLLM build_app eagerly constructs middleware_stack before it returns;
    # Starlette therefore rejects add_middleware even though serving has not
    # begun. Wrap the existing stack during _add_custom_endpoints, before
    # init_app_state/uvicorn, retaining every native auth/error middleware.
    if getattr(app.state, 'server', None) is not None:
        raise ParityError('Refusing to install weight gate on an already serving app')
    from starlette.middleware.base import BaseHTTPMiddleware
    app.middleware_stack = BaseHTTPMiddleware(app.middleware_stack or app.build_middleware_stack(),
                                              dispatch=prevent_unguarded_update)
