"""Local HTTP only: real transport timeout/retry/receipt regressions."""
import asyncio
import json
import time
from pathlib import Path
import sys

import pytest
from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))
import grading_judge as judge


async def serve(handler):
    app = web.Application()
    app.router.add_post('/grade', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    return runner, f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/grade'


def answer():
    return web.json_response({'candidates': [{'content': {'parts': [{'text': 'BAND: 3'}]},
                                             'finishReason': 'STOP'}]})


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['500', 'timeout'])
async def test_immediate_retry_and_durable_errors(tmp_path, monkeypatch, failure):
    monkeypatch.delenv('S12_GRADING_DEADLINE_UNIX', raising=False)
    calls = []
    async def handler(request):
        calls.append(time.monotonic())
        if len(calls) == 1:
            if failure == 'timeout':
                await asyncio.sleep(.15)
            else:
                return web.json_response({'error': {'code': 500, 'message': 'test-key'}}, status=500)
        return answer()
    runner, url = await serve(handler)
    token = judge._RECEIPT.set(tmp_path/'attempt')
    try:
        out, error = await judge._gemini_request({}, model='test', key='test-key',
                                                 timeout=.04, budget=.2, url=url)
        assert out == 'BAND: 3' and not error
        assert len(calls) == 2 and calls[1] - calls[0] < .12
        log = (tmp_path/'attempt.http.jsonl').read_text()
        rows = [json.loads(x) for x in log.splitlines()]
        assert len(rows) == 4 and [r['event'] for r in rows] == ['started', 'finished']*2
        assert 'test-key' not in log
        assert rows[1]['retryable'] is True
    finally:
        judge._RECEIPT.reset(token)
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['400', '429', 'prohibited'])
async def test_terminal_error_or_retry_after_does_not_loop(monkeypatch, kind):
    monkeypatch.delenv('S12_GRADING_DEADLINE_UNIX', raising=False)
    calls = []
    async def handler(request):
        calls.append(1)
        if kind == 'prohibited':
            return web.json_response({'promptFeedback': {'blockReason': 'PROHIBITED_CONTENT'}})
        return web.json_response({'error': {'code': int(kind)}}, status=int(kind),
                                 headers={'Retry-After': '99'} if kind == '429' else {})
    runner, url = await serve(handler)
    try:
        out, error = await judge._gemini_request({}, model='test', key='key', budget=.2, url=url)
        assert not out and error and len(calls) == 1
        if kind == 'prohibited':
            assert error.startswith('No candidates: ') and 'PROHIBITED_CONTENT' in error
        if kind == '429':
            assert 'remaining grading budget' in error
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_total_deadline_covers_body_and_retries(monkeypatch):
    calls = []
    async def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return web.Response(status=503)
        await asyncio.sleep(.2)
        return answer()
    runner, url = await serve(handler)
    monkeypatch.setenv('S12_GRADING_DEADLINE_UNIX', str(time.time() + .06))
    started = time.monotonic()
    try:
        out, error = await judge._gemini_request({}, model='test', key='key', timeout=1, budget=1, url=url)
        assert not out and error and len(calls) == 2
        assert time.monotonic() - started < .15
    finally:
        await runner.cleanup()


def test_expired_phase_does_not_launch_child(monkeypatch):
    import s12_task
    monkeypatch.setenv('S12_GRADING_DEADLINE_UNIX', str(time.time() - 1))
    monkeypatch.setattr(s12_task, '_run_bounded', lambda *a, **k: pytest.fail('expired phase started'))
    result = s12_task._run_judge_phase(['unused'], phase='s3_judge')
    assert result.returncode == 124 and 'deadline' in result.stderr
