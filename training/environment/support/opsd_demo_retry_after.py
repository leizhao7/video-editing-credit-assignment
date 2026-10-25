"""Honor trusted HTTP/Google retry metadata without changing model requests."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import re


def retry_after_seconds(header, body, status, *, now=None):
    """Return a finite capped delay consumed by the existing three-attempt client.

    Google puts its minute-quota reset in google.rpc.RetryInfo, often without a
    Retry-After header. Never infer delays from generated candidate text.
    """
    delays = []
    try:
        value = float(header)
    except (TypeError, ValueError):
        try:
            at = parsedate_to_datetime(str(header))
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
            value = (at - (now or datetime.now(timezone.utc))).total_seconds()
        except (TypeError, ValueError, OverflowError):
            value = 0.
    if math.isfinite(value) and value >= 0:
        delays.append(value)
    error = body.get("error", {}) if isinstance(body, dict) else {}
    if isinstance(error, dict):
        for detail in error.get("details", []):
            if not isinstance(detail, dict) or detail.get("@type") != "type.googleapis.com/google.rpc.RetryInfo":
                continue
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and re.fullmatch(r"\d+(?:\.\d{1,9})?s", delay):
                value = float(delay[:-1])
                if math.isfinite(value):
                    delays.append(value)
    wait = max(delays, default=0.)
    if status == 429:
        # A missing delay must not create a 1s/2s retry storm. The confirmed
        # failure is a per-minute token quota; +1s covers duration rounding.
        wait = wait + 1. if wait > 0 else 60.
    return min(60., max(0., wait))
