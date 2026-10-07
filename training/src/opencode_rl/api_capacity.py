"""Retry rejected HTTP requests only; never replay a partially delivered stream."""
import asyncio
from contextlib import asynccontextmanager
from email.utils import parsedate_to_datetime
import random
import time


def retry_delay(header, index, now=None):
    now = time.time() if now is None else now
    try:
        value = float(header)
    except (TypeError, ValueError):
        try:
            value = parsedate_to_datetime(header).timestamp() - now
        except (TypeError, ValueError, AttributeError):
            value = min(60, 5 * 2 ** index)
    return max(1, min(180, value))


_cooldown_until = 0.0


@asynccontextmanager
async def api_response(client, endpoint, *, headers, data, journal, request_id):
    global _cooldown_until
    for index in range(5):
        while _cooldown_until > time.monotonic():
            await asyncio.sleep(_cooldown_until - time.monotonic())
        async with client.post(endpoint, headers=headers, data=data, allow_redirects=False) as response:
            if response.status not in (429, 500, 502, 503, 504) or index == 4:
                yield response
                return
            raw = await response.read()
            delay = retry_delay(response.headers.get('Retry-After'), index) + random.uniform(0, 1)
            journal.event('api_http_retry', {'id': request_id, 'status': response.status,
                'retry': index + 1, 'wait_s': delay, 'body': journal.store.blob(raw)})
            _cooldown_until = max(_cooldown_until, time.monotonic() + delay)
