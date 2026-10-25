"""Shared input-token admission, not a change to demonstrations or training.

Budget is from retained Google QuotaFailure evidence:20M input tokens/minute.
18M/min replenishment plus a1M burst leaves headroom; a cold instance starts
empty. CountTokens covers the full original multimodal request, not characters
or Qwen tokens. Every generateContent attempt still belongs to the native client.
"""
import asyncio
from collections import deque
import math
import time
import uuid

APP = "editing-training"  # Deployment wrapper owns the application name.
RATE_PER_MINUTE = 18_000_000
CAPACITY = 1_000_000
MAX_QUEUE_SECONDS = 35
MAX_COUNT_SECONDS = 15


def count_request(model, payload):
    return {"generateContentRequest": {"model": "models/" + model, **payload}}


def reservation_tokens(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Missing/invalid Gemini CountTokens result")
    # The count endpoint and generation can differ slightly in protocol overhead.
    cost = math.ceil(value * 1.05) + 128
    if cost > CAPACITY:
        raise ValueError("One prompt exceeds the bounded token-admission capacity")
    return cost


class TokenBucket:
    """Single-owner state; methods contain no await and are event-loop atomic."""

    def __init__(self, *, now=0., rate=RATE_PER_MINUTE / 60, capacity=CAPACITY):
        self.rate, self.capacity = rate, capacity
        self.tokens, self.updated, self.cooldown = 0., now, now
        self.grants, self.reserved_total, self.quota_errors = 0, 0, 0
        self.outstanding = {}

    def refill(self, now):
        if not math.isfinite(now) or now < self.updated:
            raise ValueError("Non-monotonic quota clock")
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        # Unknown completion remains conservatively charged. Bound lease memory.
        for key, (_, started) in list(self.outstanding.items()):
            if now - started > 600:
                self.outstanding.pop(key)

    def delay(self, cost, now):
        if type(cost) is not int or not 0 < cost <= self.capacity:
            raise ValueError("Invalid token reservation")
        self.refill(now)
        return max(0., self.cooldown - now, (cost - self.tokens) / self.rate)

    def take(self, key, cost, now):
        if key in self.outstanding or self.delay(cost, now) > 1e-8:
            raise ValueError("Premature or duplicate quota admission")
        self.tokens -= cost
        self.grants += 1
        self.reserved_total += cost
        self.outstanding[key] = (cost, now)

    def observe(self, key, status, actual, retry_after, now):
        self.refill(now)
        reservation = self.outstanding.pop(key, None)
        if reservation is None:
            raise ValueError("Unknown/already completed quota lease")
        if type(actual) is int and actual > reservation[0]:
            self.tokens -= actual - reservation[0]
        # Do not refund: unknown/network failures may already have consumed quota.
        if status == 429:
            delay = float(retry_after)
            if not math.isfinite(delay) or not 0 < delay <= 60:
                delay = 60.
            self.cooldown = max(self.cooldown, now + delay)
            self.tokens = min(self.tokens, 0.)
            self.quota_errors += 1


class Admission:
    def __init__(self, *, clock=time.monotonic, sleep=asyncio.sleep, bucket=None):
        self.clock, self.sleep = clock, sleep
        self.bucket = bucket or TokenBucket(now=clock())
        self.waiting = deque()

    async def acquire(self, count, timeout=MAX_QUEUE_SECONDS):
        cost = reservation_tokens(count)
        if not 0 < timeout <= MAX_QUEUE_SECONDS:
            raise ValueError("Unbounded quota queue wait")
        key, started = uuid.uuid4().hex, self.clock()
        self.waiting.append(key)
        try:
            async with asyncio.timeout(timeout):
                while True:
                    now = self.clock()
                    delay = self.bucket.delay(cost, now)
                    if self.waiting[0] == key and delay <= 1e-8:
                        self.bucket.take(key, cost, now)
                        self.waiting.popleft()
                        return {"lease": key, "counted_input_tokens": count,
                                "reserved_input_tokens": cost, "queue_seconds": now - started}
                    await self.sleep(min(.25, max(.01, delay)))
        finally:
            if key in self.waiting:
                self.waiting.remove(key)

    def describe(self):
        self.bucket.refill(self.clock())
        return {"rate_per_minute": self.bucket.rate * 60, "capacity": self.bucket.capacity,
                "available_tokens": self.bucket.tokens, "waiting": len(self.waiting),
                "outstanding": len(self.bucket.outstanding), "grants": self.bucket.grants,
                "reserved_total": self.bucket.reserved_total, "quota_errors": self.bucket.quota_errors,
                "cooldown_seconds": max(0., self.bucket.cooldown - self.clock())}
