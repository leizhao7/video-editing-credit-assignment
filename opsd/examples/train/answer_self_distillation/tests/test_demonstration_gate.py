from examples.train.answer_self_distillation.tests.source_fixtures import source_root
import asyncio
import ast
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS

import pytest

from examples.train.answer_self_distillation import demonstration_gate as gate
from examples.train.answer_self_distillation.turn_demonstrations import (
    DemonstrationRefused, TurnDemonstrations, defaults, refusal_reason,
)
from examples.train.answer_self_distillation.trainer import AnswerOpsdTrainerMixin
from examples.train.answer_self_distillation.tests.test_turn_demonstrations import (
    TextTokenizer, journal_fixture, raw_response,
)


def setup(tmp_path, monkeypatch, transport, concurrency=1):
    data, jobs, row, _ = journal_fixture(tmp_path)
    plan = {"run_root": str(tmp_path), "opsd": {
        "demonstrator": {**defaults(), "concurrency": concurrency}, "max_hint_tokens": 2048}}
    service = TurnDemonstrations(plan, TextTokenizer(), transport, sleep=lambda _: None)
    monkeypatch.setattr(gate, "TurnDemonstrations", lambda *_: service)
    admission = gate.DemonstrationAdmission(plan, TextTokenizer())
    sequence = jobs[0]["context"].sequence
    prompt = jobs[0]["context"].span["start"]
    output = {"ocrl_rows": [copy.deepcopy(row)], "prompt_token_ids": [sequence[:prompt]],
              "response_ids": [sequence[prompt:]], "loss_masks": [[1, 1]], "rewards": [0.]}
    return admission, output, service, jobs


def test_refusal_is_trusted_metadata_not_model_text():
    assert refusal_reason({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}) == "PROHIBITED_CONTENT"
    assert refusal_reason({"candidates": [{"finishReason": "SAFETY"}]}) == "SAFETY"
    assert refusal_reason(raw_response({"text": "PROHIBITED_CONTENT", "calls": []})) is None
    assert refusal_reason({"candidates": [{"finishReason": "MAX_TOKENS"}]}) is None


def test_success_qualifies_exact_journals_without_changing_student(tmp_path, monkeypatch):
    calls = []
    def transport(request):
        calls.append(request)
        return {"status": 200, "body": raw_response()}
    admission, output, service, jobs = setup(tmp_path, monkeypatch, transport)
    before = copy.deepcopy(output)
    async def run():
        try:
            assert await admission.qualify(output) is output
        finally:
            await admission.close()
    asyncio.run(run())
    proof = output["ocrl_rows"][0].pop("opsd_demo_admission")
    assert output == before and proof["ready"] and proof["refusal"] is None
    assert proof["request_ids"] == [j["context"].span["request_id"] for j in jobs[:2]]
    assert len(calls) == 2
    # Teacher's later cache read must not call Gemini again.
    service.one(jobs[0]["context"])
    assert len(calls) == 2


def test_refusal_excludes_whole_trajectory_and_stops_unsent_siblings(tmp_path, monkeypatch):
    calls = []
    def transport(request):
        calls.append(request)
        return {"status": 200, "body": {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}}
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
    assert proof["refusal"]["whole_trajectory_excluded"]
    assert proof["refusal"]["model_reward_zero"] is False
    assert len(calls) == 1  # Neither repeated refusal requests nor remaining-turn RPCs.
    assert len(list(tmp_path.glob("opsd_demo_exclusions/*.json"))) == 1
    attempts = list(tmp_path.glob("opsd_demonstrations/*/attempt_*.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_bytes())["response"]["body"]["promptFeedback"]["blockReason"] == "PROHIBITED_CONTENT"


def test_other_exhausted_failure_is_not_silently_filtered(tmp_path, monkeypatch):
    # Authentication/configuration errors remain fatal; transient503 now has
    # separate tests for bounded whole-trajectory exclusion.
    admission, output, _, _ = setup(tmp_path, monkeypatch, lambda _: {"status": 401, "body": {}})
    async def run():
        try:
            with pytest.raises(RuntimeError, match="no teacher fallback"):
                await admission.qualify(output)
        finally:
            await admission.close()
    asyncio.run(run())
    assert "opsd_demo_admission" not in output["ocrl_rows"][0]
    assert not list(tmp_path.glob("opsd_demo_exclusions/*.json"))


def test_shared_api_limit_covers_concurrent_callers(tmp_path, monkeypatch):
    active, maximum, lock = 0, 0, threading.Lock()
    def transport(_):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(active, maximum)
        time.sleep(.015)
        with lock:
            active -= 1
        return {"status": 200, "body": raw_response()}
    admission, _, service, jobs = setup(tmp_path, monkeypatch, transport, concurrency=2)
    contexts = []
    for i in range(8):
        ctx = copy.deepcopy(jobs[0]["context"])
        ctx.span["request_id"] = f"request-{i}"
        contexts.append(ctx)
    async def run():
        try:
            await asyncio.gather(*(asyncio.to_thread(service.one, ctx) for ctx in contexts))
        finally:
            await admission.close()
    asyncio.run(run())
    assert maximum == 2


@pytest.mark.parametrize("reason", ["PROHIBITED_CONTENT", "invalid_demonstration_output",
                                   "demonstration_transport_unavailable"])
def test_native_sampler_filters_one_and_refills_to_sixteen(reason):
    # Execute the unmodified native methods without importing the GPU trainer's
    # optional torchdata/Ray runtime into the CPU-only regression environment.
    from tqdm import tqdm
    source = source_root() / "skyrl/train/fully_async_trainer.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FullyAsyncRayPPOTrainer")
    names = {"_drain_next_group", "_collect_generation_mini_batch"}
    methods = [n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *methods], type_ignores=[])
    namespace = {"asyncio": asyncio, "tqdm": tqdm}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    native = type("NativeCollector", (), {name: namespace[name] for name in names})
    class Trainer(AnswerOpsdTrainerMixin, native):
        pass
    trainer = Trainer.__new__(Trainer)
    trainer.opsd_plan = {"opsd": {"privileged_information": "gemini_turn_action", "demonstrator": defaults()}}
    trainer._check_ocrl_configuration = lambda: None
    trainer.private_answers = NS(get=lambda _: {})
    trainer.global_step, trainer.max_staleness_steps = 1, 1
    trainer.sample_full_batch, trainer.mini_batch_size = True, 16
    trainer.all_metrics, trainer.all_timings = {}, {}
    trainer._phase_gauge = NS(timed_phase=lambda *_: nullcontext())
    filtered, released = [], []
    async def mark(uids): filtered.extend(uids)
    async def release(): released.append(True)
    trainer.async_train_dataloader = NS(mark_filtered_uids=mark)
    trainer._staleness_manager = NS(on_rollout_filtered=release)
    async def run():
        queue, done = asyncio.Queue(), asyncio.Event()
        for i in range(17):
            uid, tid, rid = f"task-{i}", f"trajectory-{i}", f"request-{i}"
            row = {"task_id": uid, "trajectory_id": tid, "causal_spans": [{"request_id": rid}],
                   "opsd_demo_admission": {"format": gate.FORMAT, "request_ids": [rid], "ready": i > 0,
                       "refusal": None if i else {"whole_trajectory_excluded": True, "task": uid,
                           "trajectory_id": tid, "reason": reason,
                           "output_attempts": 3, "last_error_type": (
                               "DemonstrationTransportError" if reason == "demonstration_transport_unavailable"
                               else "DemonstrationOutputError")}}}
            queue.put_nowait(NS(uid=uid, global_step_when_scheduled=1,
                generator_output={"loss_masks": [[1]], "rewards": [0.], "ocrl_rows": [row]}))
        done.set()
        return await trainer._collect_generation_mini_batch(queue, done)
    kept, dropped, exhausted = asyncio.run(run())
    assert len(kept) == 16 and len(dropped) == 1 and not exhausted
    assert filtered == ["task-0"] and len(released) == 1
    assert all(g.generator_output["rewards"] == [0.] for g in kept)


def test_qualified_generator_reuses_rollout_and_validation_methods():
    from examples.train.answer_self_distillation.generator import AnswerOpsdGenerator
    qualified = gate.qualified_generator_type(AnswerOpsdGenerator)
    for name in ("_one", "_completed_rollout", "evaluation_one"):
        assert getattr(qualified, name) is getattr(AnswerOpsdGenerator, name)
