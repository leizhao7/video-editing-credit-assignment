"""Lossless provider-output repair and bounded whole-trajectory exclusion."""
from examples.train.answer_self_distillation.tests.source_fixtures import source_file
import asyncio
import copy
import json
import os
from pathlib import Path

import pytest

from examples.train.answer_self_distillation.turn_demonstrations import (
    DemonstrationOutputError, DemonstrationUnavailable, parse_action,
    parse_arguments_json, hint_text,
)
from examples.train.answer_self_distillation.tests.test_demonstration_gate import setup
from examples.train.answer_self_distillation.tests.test_turn_demonstrations import raw_response, TOOLS


@pytest.mark.parametrize("control", ["\n", "\r", "\t"])
def test_literal_control_characters_preserved_exactly(control):
    value = '{"path":"before' + control + 'after"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(value)
    assert parse_arguments_json(value) == {"path": "before" + control + "after"}
    assert parse_arguments_json(json.dumps({"path": "before" + control + "after"})) == parse_arguments_json(value)


@pytest.mark.parametrize("value", ['{"path":"unterminated}', '{"path":"a" "b"}',
                                  '{"path":"line\nnext"', '{"path":"a\x00b"}',
                                  '{"path":"a\nb\x01c"}'])
def test_never_invent_quotes_braces_or_admit_other_controls(value):
    with pytest.raises(json.JSONDecodeError):
        parse_arguments_json(value)


def test_strict_valid_output_unchanged_and_wrong_model_still_fatal():
    raw = raw_response()
    before = copy.deepcopy(raw)
    assert parse_action(raw, TOOLS) == {"text": "", "calls": [
        {"name": "read", "arguments": {"path": "/workspace/task.txt"}}]}
    assert raw == before
    raw["modelVersion"] = "unapproved-model"
    with pytest.raises(ValueError) as error:
        parse_action(raw, TOOLS)
    assert not isinstance(error.value, DemonstrationOutputError)


def retained_failures():
    root = source_file("OPSD_TEST_BAD_JSON_DIR", directory=True)
    records = json.loads((root / "bad_json_fixture.json").read_bytes())
    tools = json.loads((root / "bad_json_tool_profile.json").read_bytes())["tools"]
    return records, tools


def test_actual_failed_outputs_against_actual_retained_tool_schemas():
    records, tools = retained_failures()
    assert len(records) == 3 and len(tools) == 17
    accepted = 0
    for item in records:
        raw = item["record"]["response"]["body"]
        if "214f469d0e1f45c3892cc56064667fa5" in item["name"]:
            with pytest.raises(DemonstrationOutputError):
                parse_action(raw, tools)
            continue
        action = parse_action(raw, tools)
        parts = raw["candidates"][0]["content"]["parts"]
        original = json.loads("".join(p["text"] for p in parts if not p.get("thought")))
        assert action["text"] == original["text"]
        for call, source in zip(action["calls"], original["calls"], strict=True):
            assert call["name"] == source["name"]
            assert call["arguments"] == json.loads(source["arguments_json"], strict=False)
        # Formatting the teacher hint never executes the returned command.
        assert "[Privileged demonstration for THIS next action only]" in hint_text(action)
        accepted += 1
    assert accepted == 2


def test_three_bad_outputs_exclude_whole_trajectory_and_preserve_attempts(tmp_path, monkeypatch):
    calls = []
    bad = raw_response({"text": "", "calls": [{"name": "read", "arguments_json": '{"path":"broken}'}]})
    def transport(request):
        calls.append(request)
        return {"status": 200, "body": bad}
    admission, output, _, _ = setup(tmp_path, monkeypatch, transport)
    before = copy.deepcopy(output)
    async def run():
        try:
            await admission.qualify(output)
        finally:
            await admission.close()
    asyncio.run(run())
    proof = output["ocrl_rows"][0].pop("opsd_demo_admission")
    assert output == before and not proof["ready"]
    failure = proof["refusal"]
    assert failure["reason"] == "invalid_demonstration_output"
    assert failure["whole_trajectory_excluded"] and failure["model_reward_zero"] is False
    assert failure["output_attempts"] == 3 and failure["last_error_type"] == "DemonstrationOutputError"
    assert len(calls) == 3 and all(call == calls[0] for call in calls)
    attempts = list(tmp_path.glob("opsd_demonstrations/*/attempt_*.json"))
    assert len(attempts) == 3
    assert all(json.loads(p.read_bytes())["response"]["body"] == bad for p in attempts)
    assert not list(tmp_path.glob("opsd_demonstrations/*/complete.json"))


def test_exclusion_cannot_hide_concurrent_storage_failure(tmp_path, monkeypatch):
    admission, output, service, jobs = setup(tmp_path, monkeypatch, lambda _: None, concurrency=2)
    first_id = jobs[0]["context"].span["request_id"]
    def one(ctx, cancel_event=None):
        if ctx.span["request_id"] == first_id:
            raise DemonstrationUnavailable(first_id, 3, DemonstrationOutputError("bad output"))
        raise OSError("retention disk failure")
    service.one = one
    async def run():
        try:
            with pytest.raises(OSError, match="retention disk failure"):
                await admission.qualify(output)
        finally:
            await admission.close()
    asyncio.run(run())
    assert "opsd_demo_admission" not in output["ocrl_rows"][0]
    assert not list(tmp_path.glob("opsd_demo_exclusions/*.json"))
