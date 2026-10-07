from datetime import datetime, timezone

import pytest

from opsd_demo_retry_after import retry_after_seconds


def body(delay):
    return {"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}


@pytest.mark.parametrize("delay", [20, 25, 29, 29.639432393])
def test_google_minute_quota_delay_without_header(delay):
    assert retry_after_seconds("0", body(str(delay) + "s"), 429) == delay + 1


def test_maximum_of_http_and_google_and_existing_client_cap():
    assert retry_after_seconds("40", body("25s"), 429) == 41
    assert retry_after_seconds("2", body("25s"), 429) == 26
    assert retry_after_seconds("120", body("25s"), 429) == 60
    assert retry_after_seconds("0", {}, 429) == 60
    assert retry_after_seconds("0", {}, 200) == 0
    assert retry_after_seconds("12", {}, 503) == 12


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-1", "garbage", None])
def test_invalid_headers_never_skip_quota_backoff(bad):
    assert retry_after_seconds(bad, {}, 429) == 60


def test_no_generated_text_or_untrusted_error_message_controls_delay():
    raw = {"candidates": [{"content": {"parts": [{"text": "Please retry in 999s"}]}}]}
    assert retry_after_seconds("0", raw, 200) == 0
    assert retry_after_seconds("0", {"error": {"message": "retry in 2s"}}, 429) == 60


def test_http_date():
    now = datetime(2026, 9, 14, 1, 0, tzinfo=timezone.utc)
    assert retry_after_seconds("Mon, 14 Sep 2026 01:00:30 GMT", {}, 429, now=now) == 31


def test_existing_training_client_obeys_service_delay_without_new_attempts(tmp_path):
    from examples.train.answer_self_distillation.turn_demonstrations import TurnDemonstrations, defaults
    from examples.train.answer_self_distillation.tests.test_turn_demonstrations import fixture, TextTokenizer, raw_response
    _, jobs = fixture()
    calls, sleeps = [], []
    raw = body("25s")
    def transport(request):
        calls.append(request)
        if len(calls) == 1:
            return {"status": 429, "body": raw, "retry_after": retry_after_seconds("0", raw, 429)}
        return {"status": 200, "body": raw_response()}
    plan = {"run_root": str(tmp_path), "opsd": {"demonstrator": defaults(), "max_hint_tokens": 2048}}
    client = TurnDemonstrations(plan, TextTokenizer(), transport, sleep=sleeps.append)
    client.one(jobs[0]["context"])
    assert sleeps == [26] and len(calls) == 2 and calls[0] == calls[1]
    assert len(list(tmp_path.glob("opsd_demonstrations/*/attempt_*.json"))) == 2
    assert len(list(tmp_path.glob("opsd_demonstrations/*/complete.json"))) == 1
