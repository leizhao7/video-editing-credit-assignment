"""Single shared train/eval controller around the isolated native tool host."""
import asyncio
import json
import uuid

from opencode_rl.contract import OPENCODE_COMMIT, ParityError, canonical, digest
from opencode_rl.capacity import durable_io, journal_object
from opencode_rl.music_observation import validate_catalog
from .client import AsyncJournal, connection
from .loop import Limits, rollout
from .renderer import SuffixRenderer
from .budget_notice import BudgetNotices, initial_profile


class ToolSession:
    def __init__(self, process, journal):
        self.process, self.journal = process, journal
        self.profile = asyncio.get_running_loop().create_future()
        self.pending = {}
        self.exit = None
        self.reader = asyncio.create_task(self.read())

    async def read(self):
        error = None
        try:
            buffer = ''
            async for chunk in self.process.stdout:
                buffer += chunk.decode() if isinstance(chunk, bytes) else chunk
                if len(buffer) > 100_000_000:
                    raise ParityError('Oversized native tool transport frame')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    frame = await durable_io(json.loads, line)
                    kind = frame.get('kind')
                    await journal_object(self.journal, 'tool_host_frame', {'kind': kind}, frame)
                    if kind == 'profile':
                        profile = frame['profile']
                        if (self.profile.done() or profile.get('opencode_commit') != OPENCODE_COMMIT or
                                profile.get('native_agent_loop') is not False):
                            raise ParityError('Duplicate/unpinned tool profile')
                        self.profile.set_result(profile)
                    elif kind == 'tool_result':
                        future = self.pending.get(frame.get('id'))
                        if future is None or future.done():
                            raise ParityError('Unsolicited/duplicate tool result')
                        future.set_result(frame['result'])
                    elif kind == 'runtime_exit':
                        if self.exit is not None or frame['returncode'] != 0:
                            raise ParityError('Tool-only runtime failed')
                        self.exit = frame
                    elif kind == 'runtime_music_catalog':
                        if digest(validate_catalog(frame['catalog'])) != frame['catalog_sha256']:
                            raise ParityError('Music catalogue provenance mismatch')
                    elif kind not in ('runtime_stderr', 'runtime_media_observation', 'runtime_media_error'):
                        raise ParityError('Unknown tool-host transport frame')
            if buffer or self.exit is None:
                raise ParityError('Tool-only transport ended without a complete exit')
        except BaseException as exc:
            error = exc
            raise
        finally:
            error = error or ParityError('Tool host exited while an operation was pending')
            for future in (self.profile, *self.pending.values()):
                if not future.done():
                    future.set_exception(error)

    async def step(self, action):
        if self.pending:
            raise ParityError('Concurrent policy turns in one tool host')
        if self.reader.done():
            await self.reader
            raise ParityError('Tool host already exited')
        key = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[key] = future
        try:
            self.process.stdin.write(canonical({'id': key, 'method': 'step', 'action': action}) + b'\n')
            await self.process.stdin.drain.aio()
            return await future
        finally:
            self.pending.pop(key, None)

    async def close(self, *, successful):
        if not successful:
            try:
                self.process.stdin.write_eof()
            except (OSError, RuntimeError):
                pass
            finally:
                self.reader.cancel()
                await asyncio.gather(self.reader, return_exceptions=True)
                for future in (self.profile, *self.pending.values()):
                    if not future.done():
                        future.cancel()
                    elif not future.cancelled():
                        future.exception()  # consume setup errors even if shutdown won the race
            # Sandbox attempt owner stops policy descendants and archives.
            return
        try:
            self.process.stdin.write_eof()
            await self.process.stdin.drain.aio()
            await asyncio.wait_for(self.reader, timeout=120)
            await self.process.wait.aio()
            if self.process.returncode != 0:
                raise ParityError('Tool host failed during final disposal')
        finally:
            if not self.reader.done():
                self.reader.cancel()
                await asyncio.gather(self.reader, return_exceptions=True)


async def run(process, backend_url, journal, loop_journal, *, attempt_id, contract, request_gate=None, process_session=None):
    tools = ToolSession(process, journal)
    complete = False
    try:
        limits = Limits(context=contract.max_context, output=contract.max_output, turns=contract.max_turns)
        profile = initial_profile(await tools.profile, limits)
        await journal_object(journal, "harness_budget_profile",
            profile["harness_budget"], profile)
        budget = BudgetNotices(limits, profile["final_turn_prompt"])
        if process_session is not None:
            # broker adds the shared delivery/resource instructions. Judge the
            # exact task given to renderer, not the pre-broker bundle text.
            process_session.bind_task(profile['task'])
        async with connection(backend_url, contract, tools=profile['tools'], request_gate=request_gate) as engine:
            renderer = SuffixRenderer(engine, **await engine.markers(), final_turn_prompt=profile['final_turn_prompt'])
            initial = await renderer.initial(profile)
            result = await rollout(initial=initial, engine=engine, renderer=renderer, tools=tools,
                journal=AsyncJournal(loop_journal), attempt_id=attempt_id, session_id=profile['session_id'],
                contract_sha256=contract.sha256,
                limits=limits, budget=budget,
                budget_observer=process_session.record_harness_notice if process_session is not None else None,
                turn_observer=process_session.observe if process_session is not None else None)
        complete = True
        return result
    finally:
        await tools.close(successful=complete)
