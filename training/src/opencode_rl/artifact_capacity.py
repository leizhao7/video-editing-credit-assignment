"""Bounded artifact lanes: periodic saves cannot starve final deliveries."""
import asyncio
from contextlib import asynccontextmanager


class ArtifactCapacity:
    def __init__(self):
        # Same total eight streams as before. At least two remain available
        # to small grading inputs; at most two bulk streams are periodic.
        self.total = asyncio.Semaphore(8)
        self.bulk = asyncio.Semaphore(6)
        self.background = asyncio.Semaphore(2)
        # Acquire BEFORE launching a remote periodic snapshot. A queued save
        # can then be cancelled without draining an exec that never started.
        self.periodic_snapshots = asyncio.Semaphore(2)

    @asynccontextmanager
    async def transfer(self, lane):
        if lane == 'control':
            async with self.total:
                yield
        elif lane == 'terminal':
            async with self.bulk, self.total:
                yield
        elif lane == 'periodic':
            async with self.background, self.bulk, self.total:
                yield
        else:
            raise ValueError('Unknown artifact transfer lane')


_loop = _capacity = None


def artifact_capacity():
    global _loop, _capacity
    loop = asyncio.get_running_loop()
    if _loop is not loop:
        _loop, _capacity = loop, ArtifactCapacity()
    return _capacity
