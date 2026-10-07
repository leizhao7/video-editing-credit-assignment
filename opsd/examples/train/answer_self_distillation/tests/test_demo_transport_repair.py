import asyncio
import copy
import json
import sys
from types import SimpleNamespace as NS

import pytest

from examples.train.answer_self_distillation.turn_demonstrations import (
    DemonstrationTransportError, DemonstrationUnavailable, TurnDemonstrations,
    checked_transport_response,
)
from examples.train.answer_self_distillation.tests.test_demonstration_gate import setup
from examples.train.answer_self_distillation.tests.test_turn_demonstrations import TextTokenizer, raw_response
from examples.train.answer_self_distillation.trainer import AnswerOpsdTrainerMixin


def native_admission_trainer(output, plan):
    trainer = AnswerOpsdTrainerMixin.__new__(AnswerOpsdTrainerMixin)
    trainer.opsd_plan = copy.deepcopy(plan)
    trainer.opsd_plan["opsd"]["privileged_information"] = "gemini_turn_action"
    trainer._check_ocrl_configuration = lambda: None
    trainer.private_answers = NS(get=lambda _: {})
    trainer.global_step, trainer.max_staleness_steps = 1, 1
    group = NS(uid=output["ocrl_rows"][0]["task_id"], global_step_when_scheduled=1,
               generator_output=output)
    return trainer, group


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_only_explicit_transient_http_statuses_are_recoverable(status):
    with pytest.raises(DemonstrationTransportError):
        checked_transport_response({"status": status})


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, "429", None])
def test_configuration_and_malformed_protocol_still_fail_closed(status):
    with pytest.raises(ValueError):
        checked_transport_response({"status": status})


def test_mixed_transient_attempts_exclude_whole_trajectory_without_changing_student(tmp_path, monkeypatch):
    responses = [{"status": status, "body": {"error": {"status": "TRANSIENT_TEST"}}, "retry_after": delay}
                 for status, delay in ((429, 23), (503, 5), (504, 5))]
    calls, waits = [], []
    def transport(request):
        calls.append(copy.deepcopy(request))
        return responses[len(calls) - 1]
    admission, output, service, _ = setup(tmp_path, monkeypatch, transport)
    service.sleep = waits.append
    original = copy.deepcopy(output)
    async def run():
        try:
            await admission.qualify(output)
        finally:
            await admission.close()
    asyncio.run(run())
    proof = output["ocrl_rows"][0].pop("opsd_demo_admission")
    assert output == original and proof["ready"] is False
    assert proof["refusal"]["reason"] == "demonstration_transport_unavailable"
    assert proof["refusal"]["whole_trajectory_excluded"] and not proof["refusal"]["model_reward_zero"]
    assert proof["refusal"]["output_attempts"] == 3 and waits == [23, 5]
    assert len(calls) == 3 and all(call == calls[0] for call in calls)
    retained = [json.loads(p.read_bytes()) for p in tmp_path.glob("opsd_demonstrations/*/attempt_*.json")]
    assert len(retained) == 3 and {r["response"]["status"] for r in retained} == {429, 503, 504}
    assert not list(tmp_path.glob("opsd_demonstrations/*/complete.json"))
    # Carry the REAL upstream result through the actual downstream admission
    # method, not just the demonstration producer. The old code raised here.
    output["ocrl_rows"][0]["opsd_demo_admission"] = proof
    trainer, group = native_admission_trainer(output, service.plan)
    assert trainer._should_keep_group(group) is False
    assert output["rewards"] == original["rewards"]


@pytest.mark.parametrize("attempts,error_type", [(2, "DemonstrationTransportError"),
    (4, "DemonstrationTransportError"), (3, "ValueError"), (3, "OSError"),
    (3, "DemonstrationOutputError")])
def test_untrusted_transport_exclusion_still_fails_closed(tmp_path, monkeypatch, attempts, error_type):
    admission, output, service, _ = setup(tmp_path, monkeypatch, lambda _: {"status": 504})
    async def run():
        try:
            await admission.qualify(output)
        finally:
            await admission.close()
    asyncio.run(run())
    proof = output["ocrl_rows"][0]["opsd_demo_admission"]["refusal"]
    proof.update(output_attempts=attempts, last_error_type=error_type)
    trainer, group = native_admission_trainer(output, service.plan)
    with pytest.raises(ValueError):
        trainer._should_keep_group(group)


def test_actual_modal_poll_timeout_boundary_cancels_only_its_own_calls(tmp_path, monkeypatch):
    _, _, existing, jobs = setup(tmp_path, monkeypatch, lambda _: None)
    cancellations, timeouts, spawned = [], [], []
    class Call:
        object_id = "fc-test-demo-only"
        def get(self, timeout):
            timeouts.append(timeout)
            raise TimeoutError("generation deadline")
        def cancel(self, **kwargs):
            cancellations.append(kwargs)
    class Function:
        def spawn(self, request):
            spawned.append(copy.deepcopy(request))
            return Call()
    monkeypatch.setitem(sys.modules, "modal", NS(Function=NS(from_name=lambda *_: Function()),
                                                exception=NS(TimeoutError=TimeoutError)))
    service = TurnDemonstrations(existing.plan, TextTokenizer(), sleep=lambda _: None)
    with pytest.raises(DemonstrationUnavailable) as error:
        service.one(jobs[0]["context"])
    assert error.value.reason == "demonstration_transport_unavailable"
    assert timeouts == [210] * 3 and cancellations == [{"terminate_containers": False}] * 3
    assert len(spawned) == 3 and all(r == spawned[0] for r in spawned)


def test_transient_then_success_retains_original_action(tmp_path, monkeypatch):
    calls = []
    def transport(request):
        calls.append(request)
        return {"status": 504, "body": {"error": {}}} if len(calls) == 1 else {"status": 200, "body": raw_response()}
    _, _, service, jobs = setup(tmp_path, monkeypatch, transport)
    result = service.one(jobs[0]["context"])
    assert len(calls) == 2 and result["provenance"]["demonstrator_model"] == "gemini-3.8-flash"
