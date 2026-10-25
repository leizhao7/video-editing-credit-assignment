"""Private parent routes, mounted only in the independent append server."""
import asyncio

from opencode_rl.contract import ParityError
from .backend import Backend


def add_routes(app, store, contract, policy, *, render_concurrency=16):
    from fastapi import Request, HTTPException
    from starlette.responses import JSONResponse
    backend = None

    def get_backend():
        nonlocal backend
        if backend is None:
            backend = Backend(app.state, store, contract, policy, render_concurrency=render_concurrency)
        return backend

    async def payload(request):
        body = await request.json()
        if type(body) is not dict or body.pop('contract_sha256', '') != contract.sha256:
            raise HTTPException(409, 'Append harness identity mismatch')
        return body

    @app.post('/append/markers')
    async def markers(request: Request):
        if await payload(request):
            raise HTTPException(400, 'Unexpected marker fields')
        return get_backend().markers()

    @app.post('/append/render')
    async def render(request: Request):
        body = await payload(request)
        if set(body) != {'messages', 'tools', 'add_generation_prompt'}:
            raise HTTPException(400, 'Unexpected render fields')
        return await get_backend().render(**body)

    @app.post('/append/generate')
    async def generate(request: Request):
        body = await payload(request)
        if set(body) != {'request', 'tools'}:
            raise HTTPException(400, 'Unexpected generate fields')
        # FastAPI does not automatically cancel a nonstream response task on
        # disconnect. Abort the actual generation before releasing its lease.
        work = asyncio.create_task(get_backend().generate(**body))
        stopping = asyncio.Event()
        async def disconnected():
            # Starlette probes receive() inside an already-cancelled AnyIO
            # scope. Cancelling this task during that probe can be swallowed
            # by the inner scope, leaving the completed response in gather()
            # forever. Use cooperative shutdown; never depend on cancellation
            # of the disconnect probe to deliver a finished generation.
            while not stopping.is_set():
                if await request.is_disconnected():
                    return True
                try:
                    await asyncio.wait_for(stopping.wait(), timeout=.25)
                except TimeoutError:
                    pass
            return False
        watcher = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
            if work in done:
                return await work
            if await watcher:
                raise HTTPException(499, 'Parent controller disconnected; attempt is incomplete')
            raise ParityError('Disconnect watcher stopped before generation completed')
        finally:
            stopping.set()
            if not work.done():
                work.cancel()
            await asyncio.gather(work, watcher, return_exceptions=True)

    @app.exception_handler(ParityError)
    async def parity_error(request: Request, exc: ParityError):
        return JSONResponse(status_code=409, content={'error': str(exc)[:2000]})
