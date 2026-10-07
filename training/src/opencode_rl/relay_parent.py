"""Trusted-side transport over Modal exec; model code never runs on this host."""
from __future__ import annotations
import asyncio
import base64
import json
import time
from urllib.parse import urlparse

from .contract import ParityError
from .capacity import durable_io, journal_blob, journal_object
from .context_recovery import is_context_response


async def forward_runtime(process, backend_url, journal, *, attempt_id, contract_sha256, timeout=7200,
                          api_transport=None, request_gate=None):
    import aiohttp
    parsed = urlparse(backend_url)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ParityError("Invalid fixed inference backend")
    endpoint = backend_url.rstrip("/") + "/v1/chat/completions"
    if api_transport is not None:
        from .gemini_transport import GeminiTransport
        from .azure_gpt55_transport import AzureGPT55Transport
        if type(api_transport) not in (GeminiTransport, AzureGPT55Transport):
            raise ParityError("Unreviewed API transport")
        endpoint = api_transport.endpoint
    writes = asyncio.Lock()
    in_flight = {}
    request_roles = {}
    response_status = {}
    shutdown_title_cancels = set()
    pending_context = {}
    runtime_exit = None

    async def send(frame):
        async with writes:
            process.stdin.write(json.dumps(frame, separators=(",", ":")).encode() + b"\n")
            await process.stdin.drain.aio()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout, sock_read=600)) as client:
        async def request(frame):
            acquired = False
            try:
                headers = {key.lower(): value for key, value in frame["headers"].items()}
                if headers.get("x-ocrl-attempt") != attempt_id or headers.get("x-ocrl-contract") != contract_sha256:
                    raise ParityError("Sandbox tried to cross attempt/contract boundary")
                request_roles[frame['id']] = (headers.get('x-ocrl-agent'), headers.get('x-ocrl-primary'))
                headers["content-type"] = "application/json"
                session = headers.get('x-ocrl-session', '')
                agent, primary = request_roles[frame['id']]
                body = await durable_io(base64.b64decode, frame["body_b64"], validate=True)
                await journal_blob(journal, 'relay_request', {'id': frame['id'], 'headers': headers}, body)
                if request_gate is not None:
                    queued_at = time.monotonic()
                    await durable_io(journal.event, 'inference_queue_enter', {'id': frame['id'], 'time': time.time()})
                    await request_gate.acquire()
                    acquired = True
                    await durable_io(journal.event, 'inference_queue_exit', {'id': frame['id'],
                        'wait_seconds': time.monotonic() - queued_at, 'time': time.time()})
                if api_transport is not None:
                    body = await durable_io(api_transport.prepare, body)
                    await journal_blob(journal, 'api_request', {'id': frame['id']}, body)
                    # Auth is injected only after journaling, never sent to
                    # the policy sandbox or persisted in request headers.
                    headers.pop('authorization', None)
                    headers = {**headers, **api_transport.auth_headers}
                if api_transport is not None:
                    from .api_capacity import api_response
                    response_context = api_response(client, endpoint, headers=headers, data=body,
                                                    journal=journal, request_id=frame['id'])
                else:
                    response_context = client.post(endpoint, headers=headers, data=body, allow_redirects=False)
                async with response_context as response:
                    error_body = bytearray()
                    response_status[frame['id']] = response.status
                    await durable_io(journal.event, "relay_response_start", {"id": frame["id"], "status": response.status})
                    await send({"kind": "response_start", "id": frame["id"], "status": response.status,
                                "content_type": response.headers.get("Content-Type", "application/json")})
                    async for chunk in response.content.iter_any():
                        await journal_blob(journal, 'relay_response_chunk', {'id': frame['id']}, chunk)
                        if response.status != 200 and len(error_body) <= 65536:
                            error_body.extend(chunk[:65537-len(error_body)])
                        if api_transport is not None and response.status == 200:
                            chunk = api_transport.feed(frame["id"], chunk)
                            await journal_blob(journal, 'api_runtime_response_chunk', {'id': frame['id']}, chunk)
                        if not chunk:
                            continue
                        await send({"kind": "response_chunk", "id": frame["id"], "body_b64": base64.b64encode(chunk).decode()})
                    recoverable = (api_transport is None and session and agent == 'build' and primary == '1'
                                   and is_context_response(response.status, error_body))
                    if recoverable:
                        state = pending_context.setdefault(session, {'rejections': [], 'compaction': False})
                        state['rejections'].append(frame['id'])
                        state['compaction'] = False
                        await durable_io(journal.event, 'native_context_overflow', {'id': frame['id'], 'session': session})
                    elif response.status == 200 and session in pending_context:
                        state = pending_context[session]
                        if agent == 'compaction' and primary == '0':
                            state['compaction'] = True
                        elif agent == 'build' and primary == '1' and state['compaction']:
                            await durable_io(journal.event, 'native_context_recovered', {'id': frame['id'], 'session': session,
                                'rejected_requests': state['rejections']})
                            del pending_context[session]
                    await send({"kind": "response_end", "id": frame["id"]})
                    await durable_io(journal.event, "relay_response_end", {"id": frame["id"], "status": response.status})
                    if api_transport is not None:
                        api_transport.finish(frame["id"])
                    if response.status != 200 and not recoverable:
                        raise ParityError(f"Inference HTTP {response.status}; never assign model reward zero")
            except BaseException as exc:
                await durable_io(journal.event, "relay_error", {"request_id": frame["id"], "error": type(exc).__name__})
                if not isinstance(exc, asyncio.CancelledError):
                    # Unblock the sandbox handler immediately; otherwise an
                    # upstream failure can leave both ends waiting for minutes.
                    try:
                        await asyncio.wait_for(send({"kind": "response_error", "id": frame["id"]}), timeout=10)
                    except Exception:
                        pass
                raise
            finally:
                # Cancellation/error may occur during queueing, connect, SSE,
                # or journaling. Never leak a permit or release one not owned.
                if acquired:
                    request_gate.release()

        try:
            async with asyncio.timeout(timeout):
                buffer = ""
                async for chunk in process.stdout:
                    buffer += chunk.decode() if isinstance(chunk, bytes) else chunk
                    if len(buffer) > 100_000_000:
                        raise ParityError("Relay frame exceeds bound")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        frame = await durable_io(json.loads, line)
                        kind = frame["kind"]
                        if kind == "request":
                            if frame["id"] in in_flight:
                                raise ParityError("Duplicate relay request ID")
                            in_flight[frame["id"]] = asyncio.create_task(request(frame))
                        elif kind == "cancel":
                            task = in_flight.get(frame["id"])
                            # Native OpenCode abandons its background title on
                            # successful CLI exit. Only that exact shutdown
                            # cancellation is harmless; primary/other requests,
                            # early cancellations and HTTP errors still fail.
                            if (runtime_exit and runtime_exit.get('returncode') == 0
                                    and request_roles.get(frame['id']) == ('title', '0')
                                    and response_status.get(frame['id']) == 200
                                    and frame.get('error') in ('BrokenPipeError', 'ConnectionResetError')):
                                shutdown_title_cancels.add(frame['id'])
                            if task:
                                task.cancel()
                            await durable_io(journal.event, "runtime_cancel", frame)
                        elif kind in ("runtime_stdout", "runtime_stderr"):
                            body = await durable_io(base64.b64decode, frame["body_b64"], validate=True)
                            await journal_blob(journal, kind, {}, body, field='blob')
                        elif kind == "runtime_music_catalog":
                            from .contract import digest
                            from .music_observation import validate_catalog
                            catalog = validate_catalog(frame["catalog"])
                            if digest(catalog) != frame["catalog_sha256"]:
                                raise ParityError("Runtime music catalogue integrity mismatch")
                            await journal_object(journal, kind, {'catalog_sha256': digest(catalog)}, catalog)
                        elif kind == 'runtime_media_observation':
                            await journal_object(journal, kind, {}, frame['observation'])
                        elif kind == 'runtime_media_error':
                            await durable_io(journal.event, kind, {'error': frame['error']})
                        elif kind == "runtime_exit":
                            runtime_exit = frame
                            await durable_io(journal.event, kind, frame)
                            # CLI shutdown does not wait for the native title
                            # generator. Close its successful HTTP stream now,
                            # before the relay process/Modal stdin disappears.
                            # Waiting for a later BrokenPipe cancel frame races
                            # with process exit (Modal may raise GRPCError first).
                            # Never suppress an already-failed task, an HTTP
                            # error, or any primary/other agent request.
                            if frame.get('returncode') == 0:
                                for request_id, task in in_flight.items():
                                    if (not task.done()
                                            and request_roles.get(request_id) == ('title', '0')
                                            and response_status.get(request_id) == 200):
                                        shutdown_title_cancels.add(request_id)
                                        await durable_io(journal.event, 'auxiliary_shutdown_cancel_requested',
                                                      {'id': request_id, 'agent': 'title'})
                                        task.cancel()
                        else:
                            raise ParityError("Unknown relay control frame")
                    for task in in_flight.values():
                        if task.done() and not task.cancelled() and task.exception():
                            raise task.exception()
                if buffer.strip():
                    raise ParityError("Truncated relay frame")
                results = await asyncio.gather(*in_flight.values(), return_exceptions=True)
                for request_id, result in zip(in_flight, results):
                    if isinstance(result, asyncio.CancelledError) and request_id in shutdown_title_cancels:
                        await durable_io(journal.event, 'auxiliary_shutdown_cancel_accepted', {'id': request_id, 'agent': 'title'})
                    elif isinstance(result, BaseException):
                        raise ParityError("At least one inference transport failed/cancelled") from result
                if not runtime_exit or runtime_exit["returncode"] != 0:
                    raise ParityError("OpenCode runtime did not exit successfully")
                if pending_context:
                    raise ParityError('Native context overflow did not compact and continue successfully')
                return runtime_exit
        finally:
            for task in in_flight.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*in_flight.values(), return_exceptions=True)
