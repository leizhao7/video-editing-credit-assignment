"""Parent-only controller transport. No automatic retry of sampled actions."""
import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from opencode_rl.contract import ParityError
from opencode_rl.capacity import durable_io


logger = logging.getLogger(__name__)


class AsyncJournal:
    def __init__(self, journal):
        self.journal = journal

    async def event(self, kind, data):
        await durable_io(self.journal.event, kind, data)

    async def seal(self, status):
        await durable_io(self.journal.seal, status)


class Client:
    def __init__(self, http, url, contract, *, tools, request_gate=None):
        parsed = urlparse(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or
                parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/')):
            raise ParityError('Raw-token controller endpoints must remain host-local')
        self.http, self.url, self.contract = http, url.rstrip('/'), contract
        self.tools, self.gate = tools, request_gate

    async def call(self, action, body, *, timeout=None):
        options = {} if timeout is None else {'timeout': timeout}
        async with self.http.post(self.url + '/append/' + action,
                json={'contract_sha256': self.contract.sha256, **body}, allow_redirects=False,
                **options) as response:
            if response.status != 200:
                error = (await response.text())[:1500]
                raise ParityError(f'Append backend HTTP {response.status}: {error}')
            return await response.json()

    async def markers(self):
        import aiohttp
        # Markers are deterministic tokenizer metadata, not a sampled action.
        # Retry only connection/deadline failures. Contract/HTTP/JSON errors
        # still fail closed, and generate/render keep their original behavior.
        timeout = aiohttp.ClientTimeout(total=15, sock_connect=5)
        for attempt in range(3):
            try:
                return await self.call('markers', {}, timeout=timeout)
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                if attempt == 2:
                    raise
                delay = .25 * (attempt + 1)
                logger.warning('Append markers %s attempt %s/3 failed (%s); retrying in %.2fs',
                               self.url, attempt + 1, type(exc).__name__, delay)
                await asyncio.sleep(delay)

    async def render(self, **kwargs):
        import aiohttp
        # Rendering identical tool observations does not sample a policy or
        # execute a tool. Keep this retry separate from generate: a lost
        # sampled response must still fail the original attempt. The outer
        # trajectory deadline and cancellation continue to bound this wait.
        for attempt in range(2):
            try:
                return await self.call('render', kwargs)
            except aiohttp.ServerDisconnectedError:
                if attempt == 1:
                    raise
                logger.warning('Append render %s disconnected; retrying identical rendering once', self.url)
                await asyncio.sleep(.25)

    async def generate(self, request):
        if self.gate is None:
            return await self.call('generate', {'request': request, 'tools': self.tools})
        async with self.gate:
            return await self.call('generate', {'request': request, 'tools': self.tools})


@asynccontextmanager
async def connection(url, contract, *, tools, request_gate=None):
    import aiohttp
    # The outer task owns its whole-trajectory deadline. A single nonstream
    # request may legitimately run >10min; server capacity prevents fanout.
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)) as http:
        yield Client(http, url, contract, tools=tools, request_gate=request_gate)
