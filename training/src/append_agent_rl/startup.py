"""Bound startup request bursts without limiting active rollout concurrency."""
import asyncio
import math


class StartPacer:
    """Space outer create calls; never accumulate idle-time burst credits.

    This is per parent event loop, not a claim to reserve workspace-wide Modal
    quota. Other clients and the SDK's internal retries still share that quota.
    Keep the existing capacity retry policy for those cases.
    """

    def __init__(self, interval=.25, *, clock=None, sleep=None):
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
            raise ValueError('A positive finite startup interval is required')
        self.interval = interval
        self.clock = clock or asyncio.get_running_loop().time
        self.sleep = sleep or asyncio.sleep
        self.lock = asyncio.Lock()
        self.next_start = 0.0

    async def wait(self):
        started = self.clock()
        async with self.lock:
            # Cancellation while waiting does not reserve a future slot.
            delay = self.next_start - self.clock()
            while delay > 0:
                await self.sleep(delay)
                delay = self.next_start - self.clock()
            self.next_start = self.clock() + self.interval
        return max(0.0, self.clock() - started)


async def pace_sandbox_start():
    """4 outer create calls/s leaves margin under the observed 5/s quota.

    Share pacing between train/validation attempts and capacity retries in the
    same parent loop. Hold no lock during the remote create call or rollout.
    The loop owns the pacer, so a subsequent asyncio.run gets fresh primitives.
    """
    loop = asyncio.get_running_loop()
    pacer = getattr(loop, '_append_sandbox_start_pacer', None)
    if pacer is None:
        pacer = StartPacer()
        setattr(loop, '_append_sandbox_start_pacer', pacer)
    return await pacer.wait()
