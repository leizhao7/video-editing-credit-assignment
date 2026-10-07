"""Focused local-socket cancellation test; no model, judge, or external traffic.

The cancellation ownership scenario follows upstream uvloop PR 740. Each old
stale socket is detached after measurement so this test cannot corrupt another
connection. The candidate is the actual ResumeExperiment setup boundary.
"""
import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import patch


async def cancellation_trials(count=32):
    loop = asyncio.get_running_loop()
    server = await asyncio.start_server(lambda r, w: w.close(), '127.0.0.1', 0,
                                        family=socket.AF_INET)
    addr = server.sockets[0].getsockname()
    stale = cancelled = 0
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setblocking(False)
            await loop.sock_connect(sock, addr)
            task = loop.create_task(loop.create_connection(asyncio.Protocol, sock=sock))
            await asyncio.sleep(0)
            task.cancel()
            try:
                transport, _ = await task
            except asyncio.CancelledError:
                cancelled += 1
                if sock.fileno() >= 0:
                    stale += 1
                    sock.detach()  # libuv already owns/closes this FD: never close twice.
            else:
                transport.close()
            await asyncio.sleep(0.002)
        return {'trials': count, 'cancelled': cancelled, 'stale_socket_owners': stale,
                'loop_module': type(loop).__module__, 'loop_class': type(loop).__name__}
    finally:
        server.close()
        await server.wait_closed()


async def client_roundtrip():
    from aiohttp import web
    from append_agent_rl.client import connection
    seen = []
    async def receive(request):
        body = await request.json()
        seen.append((request.path, body))
        return web.json_response(body)
    app = web.Application()
    app.router.add_post('/append/{action}', receive)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    contract = SimpleNamespace(sha256='f' * 64)
    try:
        async with connection(f'http://127.0.0.1:{port}', contract,
                              tools=[{'name':'unchanged'}], request_gate=asyncio.Semaphore(8)) as client:
            async def one(index):
                payload = {'index': index, 'prompt_token_ids': [1, 2, 3], 'features': []}
                result = await client.generate(payload)
                assert result == {'contract_sha256': contract.sha256, 'request': payload,
                                  'tools': [{'name':'unchanged'}]}
            await asyncio.gather(*(one(i) for i in range(256)))
        assert len(seen) == 256
        return {'requests': 256, 'external_requests': 0, 'exact_payloads': True,
                'duplicate_requests': 0, 'errors': 0}
    finally:
        await runner.cleanup()


def verify():
    import uvloop
    from append_resume_entrypoint import classes
    from skyrl.train.entrypoints.main_base import BasePPOExp
    _, Experiment = classes()
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    baseline = asyncio.run(cancellation_trials())
    sentinel = object()
    with patch.object(BasePPOExp, '_setup_trainer', return_value=sentinel):
        assert Experiment._setup_trainer(object.__new__(Experiment)) is sentinel
    candidate = asyncio.run(cancellation_trials())
    assert baseline['cancelled'] == candidate['cancelled'] == 32
    assert baseline['stale_socket_owners'] > 0, 'Original cancellation defect did not reproduce'
    assert candidate['stale_socket_owners'] == 0
    assert candidate['loop_module'].startswith('asyncio.')
    transport = asyncio.run(client_roundtrip())
    return {'baseline': baseline, 'candidate': candidate, 'transport': transport,
            'actual_setup_override_exercised': True, 'gpu_started': False}
