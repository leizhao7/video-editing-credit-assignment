"""Bounded whole-trajectory engine leases, independent of per-turn admission.

Pure asyncio scheduling: no model, provider, filesystem or sampler side effects.
A lease keeps one trajectory on one engine through generation AND tool waits.
Release is synchronous on the owning event loop, so repeated cancellation cannot
leak capacity during async-context cleanup. This does not move existing caches
or guarantee cache hits; it bounds the histories competing for a cache.
"""
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
import time


def slots_from_cache(cache_tokens, max_context, *, reserve_contexts=2):
    """Reserve full contexts for allocator/cached-prefix overhead; never zero."""
    if (type(cache_tokens) is not int or cache_tokens < 1
            or type(max_context) is not int or max_context < 1
            or type(reserve_contexts) is not int or reserve_contexts < 1):
        raise ValueError('Positive integer cache/context budget required')
    slots = cache_tokens // max_context - reserve_contexts
    if slots < 1:
        raise ValueError('Cache too small for the reserved residency budget')
    return slots


@dataclass(frozen=True)
class Lease:
    serial: int
    engine_index: int
    queued_at: float
    acquired_at: float


@dataclass(eq=False)
class _Waiter:
    future: asyncio.Future
    queued_at: float
    lease: Lease | None = None


class ResidencyPool:
    def __init__(self, limits, *, max_waiters=512, clock=time.monotonic):
        if (not isinstance(limits, (list, tuple)) or not 1 <= len(limits) <= 16
                or any(type(n) is not int or not 1 <= n <= 128 for n in limits)
                or type(max_waiters) is not int or not 1 <= max_waiters <= 4096):
            raise ValueError('Invalid bounded per-engine residency limits')
        self.limits = tuple(limits)
        self.max_waiters, self.clock = max_waiters, clock
        self._loop = None
        self._queue = deque()
        self._leases = {}
        self._active = [0] * len(limits)
        self._started = [0] * len(limits)
        self._released = [0] * len(limits)
        self._high_water = [0] * len(limits)
        self._cursor = self._serial = self._max_queue = 0
        self._queue_wait_sum = self._queue_wait_max = self._held_seconds_sum = 0.
        self._cancelled_queued = self._cancelled_granted = 0

    def _bind(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise RuntimeError('Residency pool crossed event loops')
        return loop

    def _pick_engine(self):
        available = [i for i, n in enumerate(self._active) if n < self.limits[i]]
        if not available:
            return None
        # Among currently least-occupied engines, rotate ties deterministically.
        # Reservation budgets are full-context-equivalent, not request counts.
        return min(available, key=lambda i: (self._active[i]/self.limits[i],
                                             (i-self._cursor) % len(self.limits)))

    def _drain(self):
        while self._queue:
            waiter = self._queue[0]
            if waiter.future.cancelled():
                self._queue.popleft()
                continue
            index = self._pick_engine()
            if index is None:
                return
            self._queue.popleft()
            now = self.clock()
            lease = Lease(self._serial, index, waiter.queued_at, now)
            self._serial += 1
            self._cursor = (index+1) % len(self.limits)
            self._leases[lease.serial] = lease
            waiter.lease = lease
            self._active[index] += 1
            self._started[index] += 1
            self._high_water[index] = max(self._high_water[index], self._active[index])
            waited = max(0., now-waiter.queued_at)
            self._queue_wait_sum += waited
            self._queue_wait_max = max(self._queue_wait_max, waited)
            waiter.future.set_result(lease)

    async def acquire(self):
        loop = self._bind()
        if len(self._queue) >= self.max_waiters:
            raise RuntimeError('Residency waiting queue exceeded its explicit bound')
        waiter = _Waiter(loop.create_future(), self.clock())
        self._queue.append(waiter)
        self._max_queue = max(self._max_queue, len(self._queue))
        self._drain()
        try:
            return await waiter.future
        except BaseException:
            if waiter.lease is not None:
                # A free slot may be granted just before a task is cancelled,
                # before its caller ever receives the lease.
                self._cancelled_granted += 1
                self.release(waiter.lease)
            else:
                self._cancelled_queued += 1
                try:
                    self._queue.remove(waiter)
                except ValueError:
                    pass
                self._drain()
            raise

    def release(self, lease):
        self._bind()
        if not isinstance(lease, Lease) or self._leases.get(lease.serial) is not lease:
            raise ValueError('Unknown, foreign or already released residency lease')
        del self._leases[lease.serial]
        self._active[lease.engine_index] -= 1
        self._released[lease.engine_index] += 1
        self._held_seconds_sum += max(0., self.clock()-lease.acquired_at)
        self._drain()

    @asynccontextmanager
    async def lease(self):
        lease = await self.acquire()
        try:
            yield lease
        finally:
            self.release(lease)

    def snapshot(self):
        """Cheap scalar counters, never inference/queue mutation or raw data."""
        return {'limits': list(self.limits), 'active': list(self._active),
            'started': list(self._started), 'released': list(self._released),
            'high_water': list(self._high_water), 'waiting': len(self._queue),
            'max_waiting': self._max_queue, 'queue_wait_seconds_sum': self._queue_wait_sum,
            'queue_wait_seconds_max': self._queue_wait_max,
            'held_seconds_sum': self._held_seconds_sum,
            'cancelled_queued': self._cancelled_queued,
            'cancelled_after_grant': self._cancelled_granted}
