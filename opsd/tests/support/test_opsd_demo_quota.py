import asyncio
import copy

import pytest

from opsd_demo_quota import Admission, TokenBucket, count_request, reservation_tokens


def test_full_request_is_counted_without_mutation():
    payload = {"systemInstruction": {"parts": [{"text": "teacher"}]},
               "contents": [{"role": "user", "parts": [{"text": "history"},
                    {"inlineData": {"mimeType": "image/png", "data": "YWJj"}}]}],
               "generationConfig": {"responseJsonSchema": {"type": "object"}}}
    original = copy.deepcopy(payload)
    counted = count_request("gemini-3.8-flash", payload)
    assert counted == {"generateContentRequest": {"model": "models/gemini-3.8-flash", **original}}
    assert payload == original


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None, "12", 1_000_000])
def test_no_guessed_or_unbounded_token_count(value):
    with pytest.raises(ValueError):
        reservation_tokens(value)


def test_cold_start_does_not_replenish_a_fresh_full_burst():
    b = TokenBucket(now=10, rate=100, capacity=1000)
    assert b.delay(500, 10) == 5
    with pytest.raises(ValueError):
        b.take("a", 500, 10)
    b.take("a", 500, 15)
    assert b.tokens == 0 and b.grants == 1


def test_every_60_second_window_is_bounded_by_rate_plus_burst():
    b = TokenBucket(rate=300000, capacity=1000000)
    grants = []
    for i in range(18001):
        now = i / 100
        if b.delay(50000, now) == 0:
            b.take(str(i), 50000, now)
            grants.append((now, 50000))
    for end in range(60, 181):
        assert sum(n for t, n in grants if end - 60 < t <= end) <= 19_000_000


def test_feedback_is_shared_cooldown_and_is_not_refunded():
    b = TokenBucket(rate=100, capacity=1000)
    b.take("a", 500, 5)
    b.take("b", 500, 10)
    b.observe("a", 429, None, 25, 10)
    assert b.delay(100, 11) == 24
    b.observe("b", 200, 900, 0, 11)
    assert b.tokens == -300
    with pytest.raises(ValueError):
        b.observe("b", 200, 900, 0, 12)
    assert b.quota_errors == 1


def test_unknown_completion_keeps_budget_but_expires_lease_memory():
    b = TokenBucket(rate=100, capacity=1000)
    b.take("a", 500, 5)
    assert b.tokens == 0
    b.refill(606)
    assert b.outstanding == {}


def test_cancellation_removes_waiter_and_does_not_admit():
    async def scenario():
        paused = asyncio.Event()
        async def sleep(_):
            paused.set()
            await asyncio.Future()
        gate = Admission(clock=lambda: 0., sleep=sleep)
        task = asyncio.create_task(gate.acquire(100))
        await paused.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not gate.waiting and gate.bucket.grants == 0
    asyncio.run(scenario())


def test_fifo_and_no_quota_oversubscription():
    async def scenario():
        now, order = [0.], []
        async def sleep(dt):
            now[0] += dt
            await asyncio.sleep(0)
        gate = Admission(clock=lambda: now[0], sleep=sleep)
        async def one(i):
            result = await gate.acquire(10000)
            order.append(i)
            return result
        results = await asyncio.gather(*(one(i) for i in range(64)))
        assert order == list(range(64)) and not gate.waiting
        assert len({r["lease"] for r in results}) == 64
        assert gate.bucket.reserved_total == 64 * reservation_tokens(10000)
        assert gate.bucket.tokens >= -1e-6
    asyncio.run(scenario())


def test_queue_timeout_is_bounded():
    async def scenario():
        gate = Admission()
        with pytest.raises(TimeoutError):
            await gate.acquire(900000, timeout=.001)
        assert not gate.waiting and not gate.bucket.outstanding
    asyncio.run(scenario())
