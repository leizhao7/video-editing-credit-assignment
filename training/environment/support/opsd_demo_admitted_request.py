"""One exact Gemini generation attempt with bounded shared quota admission."""
import asyncio
import time

import httpx

from opsd_demo_quota import MAX_COUNT_SECONDS, MAX_QUEUE_SECONDS, count_request
from opsd_demo_retry_after import retry_after_seconds


def read_response(response):
    try:
        body = response.json()
    except ValueError:
        body = {"non_json_response": response.text[:8192]}
    return {"status": response.status_code, "body": body,
            "retry_after": retry_after_seconds(response.headers.get("retry-after", "0"), body, response.status_code)}


async def generate_once(request, body, client, gate, headers):
    """No internal retry, prompt edits, image truncation or model fallback.

Count+admission uses at most50s of the existing60s service allowance. Actual
generation retains its original150s timeout. Raw count and generation results
are returned to the existing training-side attempt journal.
    """
    started, admission = time.monotonic(), {"format": "gemini-counted-input-admission-v1"}
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/" + request["model"]
    try:
        async with asyncio.timeout(MAX_COUNT_SECONDS + MAX_QUEUE_SECONDS):
            count = await client.post(endpoint + ":countTokens", headers=headers,
                json=count_request(request["model"], request["payload"]), timeout=MAX_COUNT_SECONDS)
            counted = read_response(count)
            admission["count_response"] = counted
            admission["count_seconds"] = time.monotonic() - started
            if counted["status"] != 200:
                return {**counted, "admission": {**admission, "phase": "count_tokens", "generation_attempted": False}}
            lease = await gate.acquire(counted["body"].get("totalTokens"))
            admission.update(lease)
    except (TimeoutError, ValueError, httpx.TransportError) as error:
        return {"status": 503, "body": {"error": {"status": "LOCAL_ADMISSION_UNAVAILABLE",
                    "message": type(error).__name__ + ": " + str(error)[:500]}}, "retry_after": 5,
                "admission": {**admission, "phase": "count_or_admission", "generation_attempted": False}}
    admission["preparation_seconds"] = time.monotonic() - started
    result = None
    try:
        async with asyncio.timeout(request["timeout_seconds"]):
            response = await client.post(endpoint + ":generateContent", headers=headers, content=body,
                                         timeout=request["timeout_seconds"])
        result = read_response(response)
        return {**result, "admission": {**admission, "phase": "generated", "generation_attempted": True}}
    except (TimeoutError, httpx.TransportError) as error:
        # Return an explicit infrastructure error, not a fabricated candidate or
        # a provider refusal. The existing client owns its three retry attempts.
        # No response was obtained, so retain the full reservation conservatively.
        result = {"status": 504, "body": {"error": {"status": "DEMONSTRATOR_TRANSPORT_TIMEOUT",
                    "type": type(error).__name__, "message": str(error)[:500]}}, "retry_after": 5}
        return {**result, "admission": {**admission, "phase": "generation_transport",
                    "generation_attempted": True, "response_received": False}}
    finally:
        # A network failure may already have consumed input tokens; no refund.
        actual = (result or {}).get("body", {}).get("usageMetadata", {}).get("promptTokenCount")
        try:
            async with asyncio.timeout(5):
                state = await gate.observe(lease["lease"], (result or {}).get("status"), actual,
                                           (result or {}).get("retry_after", 0))
            print("OPSD_DEMO_QUOTA " + str({"counted": lease["counted_input_tokens"],
                "reserved": lease["reserved_input_tokens"], "actual": actual,
                "queue_seconds": lease["queue_seconds"], "status": (result or {}).get("status"),
                "quota_errors": state["quota_errors"]}), flush=True)
        except Exception as error:
            # The lease stays charged if completion feedback is lost. Preserve a
            # valid raw teacher response; monitoring sees the explicit warning.
            print("OPSD_DEMO_QUOTA_FEEDBACK_ERROR " + type(error).__name__ + ": " + str(error)[:500], flush=True)
