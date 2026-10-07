"""Serving/storage scheduling only. Does not change prompts or action semantics."""
import asyncio
import time
from contextlib import ExitStack, asynccontextmanager


async def durable_io(function, *args, **kwargs):
    """Offload fsync/hash work, but finish an accepted write before cancellation.

    Cancelling to_thread alone leaves the writer running after a journal may
    already be sealed. Shield + drain keeps archive ordering without blocking
    the shared event loop on disk I/O.
    """
    job = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(job)
    except asyncio.CancelledError:
        await job
        raise


@asynccontextmanager
async def durable_file(path, mode):
    """Open/close off-loop, including cancellation during an accepted open."""
    stack = ExitStack()
    try:
        handle = await durable_io(lambda: stack.enter_context(path.open(mode)))
        yield handle
    finally:
        await durable_io(stack.close)


async def journal_blob(journal, kind, metadata, body, field='body'):
    def write():
        journal.event(kind, {**metadata, field: journal.store.blob(body)})
    await durable_io(write)


async def journal_object(journal, kind, metadata, value, field='blob'):
    """Serialize and persist potentially large tool observations off the loop."""
    from .contract import canonical
    def write():
        journal.event(kind, {**metadata, field: journal.store.blob(canonical(value))})
    await durable_io(write)


class CoalescedCommit:
    def __init__(self, commit, interval=2):
        self.commit, self.interval = commit, interval
        self.requested = self.committed = 0
        self.task = None
        self.last = 0.0

    async def __call__(self):
        self.requested += 1
        target = self.requested
        while self.committed < target:
            if self.task is None or self.task.done():
                self.task = asyncio.create_task(self._flush())
            await asyncio.shield(self.task)

    async def _flush(self):
        # Complete one generation and release its callers. A continuously busy
        # later generation must not strand already-durable rollout/grade jobs.
        await asyncio.sleep(max(0, self.last + self.interval - time.monotonic()))
        target = self.requested
        for retry in range(5):
            try:
                await self.commit()
                break
            except Exception as exc:
                if retry == 4 or not any(text in str(exc).lower() for text in ('rate limit', 'resource_exhausted', 'failed to publish')):
                    raise
                await asyncio.sleep(min(30, 2 ** retry))
        self.committed = target
        self.last = time.monotonic()
