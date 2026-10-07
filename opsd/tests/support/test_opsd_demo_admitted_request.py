import asyncio
import json

import httpx

import pytest

from opsd_demo_admitted_request import generate_once


class Response:
    def __init__(self, status, body):
        self.status_code, self.body, self.headers = status, body, {}
    def json(self):
        return self.body


class Client:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
    async def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class Gate:
    def __init__(self):
        self.counts, self.feedback = [], []
    async def acquire(self, count):
        self.counts.append(count)
        if type(count) is not int or count <= 0:
            raise ValueError("Missing count")
        return {"lease": "one", "counted_input_tokens": count, "reserved_input_tokens": count + 128, "queue_seconds": 2}
    async def observe(self, *values):
        self.feedback.append(values)
        return {"quota_errors": int(values[1] == 429)}


def request():
    r = {"model": "gemini-3.8-flash", "payload": {"contents": [{"role": "user", "parts": [{"text": "history"}]}]},
         "timeout_seconds": 150}
    return r, json.dumps(r["payload"]).encode()


def test_one_unchanged_generation_retains_all_count_and_raw_data():
    r, body = request()
    raw = {"modelVersion": r["model"], "usageMetadata": {"promptTokenCount": 123}, "candidates": []}
    client, gate = Client([Response(200, {"totalTokens": 120}), Response(200, raw)]), Gate()
    result = asyncio.run(generate_once(r, body, client, gate, {"test": "header"}))
    assert result["body"] is raw and result["status"] == 200
    assert result["admission"]["count_response"]["body"] == {"totalTokens": 120}
    assert gate.counts == [120] and gate.feedback == [("one", 200, 123, 0)]
    assert len(client.requests) == 2 and client.requests[1][1]["content"] is body
    assert client.requests[1][1]["timeout"] == 150
    assert client.requests[0][1]["json"]["generateContentRequest"] == {"model": "models/" + r["model"], **r["payload"]}


def test_count_failure_is_retained_and_never_generates():
    r, body = request()
    raw = {"error": {"message": "quota"}}
    client, gate = Client([Response(429, raw)]), Gate()
    result = asyncio.run(generate_once(r, body, client, gate, {}))
    assert result["status"] == 429 and result["body"] is raw and result["retry_after"] == 60
    assert not result["admission"]["generation_attempted"]
    assert len(client.requests) == 1 and not gate.counts and not gate.feedback


def test_no_estimation_fallback_for_missing_count():
    r, body = request()
    client, gate = Client([Response(200, {})]), Gate()
    result = asyncio.run(generate_once(r, body, client, gate, {}))
    assert result["status"] == 503 and len(client.requests) == 1 and not gate.feedback


def test_provider_429_updates_shared_cooldown_without_hidden_retry():
    r, body = request()
    raw = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "25s"}]}}
    client, gate = Client([Response(200, {"totalTokens": 120}), Response(429, raw)]), Gate()
    result = asyncio.run(generate_once(r, body, client, gate, {}))
    assert result["status"] == 429 and result["body"] is raw
    assert len(client.requests) == 2 and gate.feedback == [("one", 429, None, 26)]


@pytest.mark.parametrize("error", [TimeoutError("network"), httpx.ReadTimeout("read"), httpx.ConnectError("connect")])
def test_network_failure_keeps_reservation_and_does_not_retry(error):
    r, body = request()
    client, gate = Client([Response(200, {"totalTokens": 120}), error]), Gate()
    result = asyncio.run(generate_once(r, body, client, gate, {}))
    assert result["status"] == 504 and "candidates" not in result["body"]
    assert result["body"]["error"]["type"] == type(error).__name__
    assert result["admission"]["generation_attempted"] and not result["admission"]["response_received"]
    assert len(client.requests) == 2 and gate.feedback == [("one", 504, None, 5)]


def test_generation_deadline_is_structured_but_cancellation_remains_cancellation():
    r, body = request()
    r["timeout_seconds"] = .001  # Exercise the actual asyncio deadline, no long test sleep.
    class SlowClient(Client):
        async def post(self, url, **kwargs):
            if url.endswith(":generateContent"):
                self.requests.append((url, kwargs))
                await asyncio.sleep(10)
            return await super().post(url, **kwargs)
    gate = Gate()
    result = asyncio.run(generate_once(r, body, SlowClient([Response(200, {"totalTokens": 120})]), gate, {}))
    assert result["status"] == 504 and result["body"]["error"]["type"] == "TimeoutError"
    assert gate.feedback == [("one", 504, None, 5)]
    client = Client([Response(200, {"totalTokens": 120}), asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(generate_once(r, body, client, Gate(), {}))


def test_programming_error_not_disguised_as_transient():
    r, body = request()
    client = Client([Response(200, {"totalTokens": 120}), ValueError("programming failure")])
    with pytest.raises(ValueError, match="programming failure"):
        asyncio.run(generate_once(r, body, client, Gate(), {}))


@pytest.mark.parametrize("status", [200, 429])
def test_quota_preemption_loses_lease_but_preserves_exact_provider_result(status, capsys):
    """Observed September15: quota actor preemption loses in-flight lease IDs.

    Feedback is accounting only: neither fabricate a generation failure nor
    silently retry/replace the actual provider response after the actor restarts.
    """
    class RestartedGate(Gate):
        async def observe(self, *values):
            self.feedback.append(values)
            raise ValueError("Unknown/already completed quota lease")

    r, body = request()
    raw = ({"modelVersion": r["model"], "usageMetadata": {"promptTokenCount": 123},
            "candidates": [{"content": {"parts": [{"text": "original demonstration"}]}}]}
           if status == 200 else {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}})
    client = Client([Response(200, {"totalTokens": 120}), Response(status, raw)])
    gate = RestartedGate()
    result = asyncio.run(generate_once(r, body, client, gate, {}))
    assert result["status"] == status and result["body"] is raw
    assert result["admission"]["phase"] == "generated"
    assert result["admission"]["generation_attempted"]
    assert result["retry_after"] == (60 if status == 429 else 0)
    assert len(client.requests) == 2 and len(gate.feedback) == 1
    assert client.requests[1][1]["content"] is body
    assert "OPSD_DEMO_QUOTA_FEEDBACK_ERROR ValueError" in capsys.readouterr().out
