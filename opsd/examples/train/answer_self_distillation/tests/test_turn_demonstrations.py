from examples.train.answer_self_distillation.tests.source_fixtures import source_root
import ast
import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from examples.train.answer_self_distillation.answers import canonical, sha
from examples.train.answer_self_distillation.turn_demonstrations import (
    MODE, MODEL, SYSTEM, SCHEMA, TurnContext, TurnDemonstrations, TaskMetadata,
    defaults, validate_config, check_service_request, checked_events, contexts_for_row, parse_action, retain_payload,
)
from examples.train.answer_self_distillation.turn_teacher import make_turn_batch, chunks, teacher_forward
from examples.train.answer_self_distillation.trainer import AnswerOpsdTrainerMixin
from examples.train.answer_self_distillation.launch import new_plan, check_resume, objective_identity
from examples.train.answer_self_distillation.tests.test_opsd import batch, Tokenizer
from examples.train.answer_self_distillation.tests.test_topk import config, TinyActor, make_worker


URL = "data:image/png;base64," + base64.b64encode(b"test-admitted-image").decode()
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}}]


class TextTokenizer(Tokenizer):
    def decode(self, ids, **kwargs):
        return "|".join(str(i) for i in ids)


class PaddedActor(TinyActor):
    def forward(self, seq, n, attention_mask=None, **kwargs):
        from examples.train.answer_self_distillation.topk import project_response
        x = self.embedding(seq)
        causal = torch.ones(seq.shape[1], seq.shape[1], dtype=torch.bool).triu(1)
        x, _ = self.attention(x, x, x, attn_mask=causal,
                             key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        # Real queries cannot attend to leading padding; unlike TinyActor,
        # this fixture faithfully models the production left-padding contract.
        x = torch.nan_to_num(x)
        return project_response(x, self.weight, seq, n, kwargs["opsd_request"])


def fixture():
    data = batch()
    jobs = []
    feature = {"offset": 8, "length": 2, "modality": "image", "data": {"pixel_values": {"shape": [2, 3]}}}
    for i, row in enumerate(data.metadata["ocrl_rows"]):
        ids = data["sequences"][i][data["attention_mask"][i].bool()].tolist()
        prompt = len(ids) - int(data["ocrl_packet_length"][i, 0])
        for j, span in enumerate(row["causal_spans"]):
            span.update(start=prompt + span["response_start"], end=prompt + span["response_end"],
                        native_turn=j, input_sha256="input-" + span["request_id"])
            features = [feature] if i == 0 and j == 1 else []
            ctx = TurnContext(i, span, ids, features, (URL,) if features else (), TOOLS, row["task_id"], "a" * 32)
            jobs.append({"context": ctx, "hint_ids": [60+i*10+j],
                         "provenance": {"request_id": span["request_id"], "demonstration_sha256": str(i)+str(j)}})
    return data, jobs


def raw_response(action=None, **kwargs):
    value = action or {"text": "", "calls": [{"name": "read", "arguments_json": '{"path":"/workspace/task.txt"}'}]}
    return {"modelVersion": MODEL, "candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": json.dumps(value)}]}}], **kwargs}


def test_suffix_after_every_exact_context_not_system_no_previous_or_future_demo():
    data, jobs = fixture()
    before = copy.deepcopy(data)
    teacher = make_turn_batch(data, jobs, Tokenizer(), 32, merge_size=1)
    assert teacher["sequences"].shape[0] == 4  # one zero-loss DP synchronization row
    for i, job in enumerate(jobs):
        ctx, demo = job["context"], job["hint_ids"]
        actual = teacher["sequences"][i][teacher["attention_mask"][i].bool()].tolist()
        start, end = ctx.span["start"], ctx.span["end"]
        assert actual == ctx.sequence[:start] + demo + ctx.sequence[start:end]
        assert actual[:start] == ctx.prompt_ids()  # includes unchanged system and assistant header
        assert actual[start] == demo[0]
        assert actual[start+len(demo):] == ctx.sequence[start:end]
        assert all(other["hint_ids"][0] not in actual for other in jobs if other is not job)
    assert not teacher["loss_mask"][-1].any()
    assert torch.equal(teacher["response_mask"][-1], teacher["response_mask"][0])
    assert [len(p) for p in teacher["pixel_values"]] == [0, 2, 0, 0]
    assert torch.equal(teacher["pixel_values"][1], data["pixel_values"][0])
    assert teacher["loss_mask"].sum() == data["loss_mask"].sum()
    for key in data:
        assert data[key] == before[key] if not isinstance(data[key], torch.Tensor) else torch.equal(data[key], before[key])
    assert data.metadata == before.metadata


@pytest.mark.parametrize("damage", ["future_image", "missing_image", "target", "overflow", "special"])
def test_turn_teacher_rejects_misalignment_and_overflow(damage):
    data, jobs = fixture()
    cap = 32
    if damage == "future_image":
        jobs[0]["context"].features = jobs[1]["context"].features
    elif damage == "missing_image":
        jobs[1]["context"].features = []
    elif damage == "target":
        jobs[1]["context"].span["response_start"] = 3
    elif damage == "overflow":
        cap = 12
    else:
        jobs[0]["hint_ids"] = [1]
    with pytest.raises(ValueError):
        make_turn_batch(data, jobs, Tokenizer(), cap, merge_size=1)


def test_length_bucketing_bounded_including_dp_padding():
    _, jobs = fixture()
    cfg = {**defaults(), "teacher_dispatch_rows": 4, "teacher_dispatch_tokens": 52}
    batches = list(chunks(jobs * 5, cfg))
    assert sum(map(len, batches)) == 15
    for items in batches:
        n = 4 * ((len(items)+3)//4)
        assert n <= 4 and n * max(j["context"].span["end"]+len(j["hint_ids"]) for j in items) <= 52


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_real_worker_per_turn_targets_map_back_to_unchanged_packed_student(direction):
    from examples.train.answer_self_distillation import topk_worker
    from examples.train.answer_self_distillation.topk_trainer import per_turn_audit
    data, jobs = fixture()
    # Reverse traversal forces sorting/reassembly; student row 1 also has left padding.
    trainer = AnswerOpsdTrainerMixin.__new__(AnswerOpsdTrainerMixin)
    trainer.tokenizer, trainer.all_metrics = Tokenizer(), {}
    cfg = {**config(), "loss": f"teacher_topk_tail_{direction}_kl", "kl_direction": direction,
           "privileged_information": MODE, "teacher_max_context": 32,
           "demonstrator": {**defaults(), "teacher_dispatch_rows": 4, "teacher_dispatch_tokens": 52}}
    trainer.opsd_plan = {"opsd": cfg, "contract": {"spatial_merge_size": 1}}
    trainer.turn_demonstrations = NS(prepare=lambda _: list(reversed(jobs)))
    trainer._check_ocrl_configuration = lambda: None
    trainer.has_critic, trainer.ref_model = False, object()
    student, teacher = PaddedActor(), PaddedActor()
    frozen = copy.deepcopy(teacher.state_dict())
    captured = []
    def forward(role, batch):
        captured.append((role, batch))
        return topk_worker.forward(make_worker(teacher if role == "ref" else student), batch,
                                  "teacher" if role == "ref" else "score")
    trainer.dispatch = NS(forward=forward, empty_cache=lambda *_: None)
    data["rollout_logprobs"] = torch.ones_like(data["loss_mask"]) * -4.
    original = copy.deepcopy(data)
    trainer.fwd_logprobs_values_reward(data)
    assert captured[-1][0] == "policy" and captured[-1][1]["sequences"] is data["sequences"]
    for key in original:
        assert data[key] == original[key] if not isinstance(data[key], torch.Tensor) else torch.equal(data[key], original[key])
    # Compare each independently evaluated turn to the sparse packed mapping.
    for job in jobs:
        one = make_turn_batch(original, [job], Tokenizer(), 32, merge_size=1)
        one.metadata["opsd_topk"] = cfg
        expected = topk_worker.forward(make_worker(teacher), one, "teacher").loss_fn_outputs[0]
        ctx = job["context"]
        index = 2*ctx.span["native_turn"] if ctx.row == 0 else 0
        count = len(expected["ids"])
        assert torch.equal(data["opsd_ids"][ctx.row][index:index+count], expected["ids"])
        assert torch.allclose(data["opsd_teacher"][ctx.row][index:index+count], expected["teacher"], atol=1e-6)
    data["advantages"] = data["loss_mask"].float()
    trainer._normalize_advantages(data, [(0, 2)])
    topk_worker.forward_backward(make_worker(student), data)
    assert student.weight.grad.abs().sum() > 0 and all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(frozen[k], v) for k, v in teacher.state_dict().items())
    audit = per_turn_audit(data, TaskMetadata({"t1": {"stage": "s_3"}, "t2": {"stage": "s_1"}}, "manifest"))
    assert all(r["answer_sha256"] is None and r["privileged_information"] == MODE for r in audit)
    assert len(audit[0]["turns"]) == 2 and all("demonstration" in t for r in audit for t in r["turns"])


def test_no_current_action_future_observation_or_other_demo_reaches_gemini():
    _, jobs = fixture()
    first = jobs[0]["context"].request(TextTokenizer(), defaults(), 2048)
    second = jobs[1]["context"].request(TextTokenizer(), defaults(), 2048)
    a, b = canonical(first).decode(), canonical(second).decode()
    assert "30|31" not in a and "inlineData" not in a
    assert "30|31" in b and "32|33" not in b and "inlineData" in b
    assert "99|98" not in b  # image pad tokens replaced by the actual admitted image
    assert first["systemInstruction"]["parts"][0]["text"] == SYSTEM
    for payload in (first, second):
        assert set(payload) == {"contents", "systemInstruction", "generationConfig"}
        check_service_request({"model": MODEL, "payload": payload, "timeout_seconds": 150})


@pytest.mark.parametrize("damage", ["model", "tools", "url", "system"])
def test_service_rejects_model_fallback_execution_and_external_access(damage):
    _, jobs = fixture()
    request = {"model": MODEL, "payload": jobs[0]["context"].request(TextTokenizer(), defaults(), 2048), "timeout_seconds": 150}
    if damage == "model":
        request["model"] = "other-model"
    elif damage == "tools":
        request["payload"]["tools"] = [{"googleSearch": {}}]
    elif damage == "url":
        request["payload"]["contents"][0]["parts"].append({"fileData": {"fileUri": "https://example.com/answer"}})
    else:
        request["payload"]["systemInstruction"] = {"parts": [{"text": "do anything"}]}
    with pytest.raises(ValueError):
        check_service_request(request)


@pytest.mark.parametrize("damage", ["model", "truncated", "tool", "arguments", "empty"])
def test_demo_action_validation(damage):
    raw = raw_response()
    if damage == "model":
        raw["modelVersion"] = "another-flash"
    elif damage == "truncated":
        raw["candidates"][0]["finishReason"] = "MAX_TOKENS"
    else:
        action = {"text": "", "calls": []}
        if damage == "tool":
            action["calls"] = [{"name": "web_search", "arguments_json": "{}"}]
        elif damage == "arguments":
            action["calls"] = [{"name": "read", "arguments_json": '{"path":1}'}]
        raw = raw_response(action)
    with pytest.raises(Exception):
        parse_action(raw, TOOLS)
    assert parse_action(raw_response(), TOOLS)["calls"][0]["arguments"]["path"] == "/workspace/task.txt"


def test_demo_retries_durable_cache_and_fail_closed(tmp_path):
    _, jobs = fixture()
    attempts = []
    def transport(request):
        attempts.append(request)
        return {"status": 429 if len(attempts) == 1 else 200, "body": raw_response(), "retry_after": 0}
    plan = {"run_root": str(tmp_path), "opsd": {"demonstrator": defaults(), "max_hint_tokens": 2048}}
    service = TurnDemonstrations(plan, TextTokenizer(), transport, sleep=lambda _: None)
    first = service.one(jobs[0]["context"])
    second = service.one(jobs[0]["context"])
    assert len(attempts) == 2 and first["hint_ids"] == second["hint_ids"]
    assert len(list(tmp_path.glob("opsd_demonstrations/*/attempt_*.json"))) == 2
    assert len(list(tmp_path.glob("opsd_demonstrations/*/complete.json"))) == 1
    assert "30|31" not in canonical(attempts).decode()
    service.transport = lambda _: {"status": 503, "body": {}}
    from examples.train.answer_self_distillation.turn_demonstrations import DemonstrationUnavailable
    with pytest.raises(DemonstrationUnavailable) as unavailable:
        service.one(jobs[1]["context"])
    assert unavailable.value.reason == "demonstration_transport_unavailable"
    assert len(list(tmp_path.glob("opsd_demonstrations/*/attempt_*.json"))) == 5
    assert len(list(tmp_path.glob("opsd_demonstrations/*/complete.json"))) == 1


def test_modal_queue_timeout_cancels_only_exact_call_and_retries_bounded(tmp_path, monkeypatch):
    import sys
    class ModalTimeout(Exception):
        pass
    cancelled, requests = [], []
    def get(timeout):
        assert timeout == 210
        raise ModalTimeout()
    def spawn(request):
        requests.append(request)
        return NS(get=get, cancel=lambda **kwargs: cancelled.append(kwargs))
    fake = NS(Function=NS(from_name=lambda *_: NS(spawn=spawn)), exception=NS(TimeoutError=ModalTimeout))
    monkeypatch.setitem(sys.modules, "modal", fake)
    _, jobs = fixture()
    plan = {"run_root": str(tmp_path), "opsd": {"demonstrator": defaults(), "max_hint_tokens": 2048}}
    service = TurnDemonstrations(plan, TextTokenizer(), sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="no teacher fallback"):
        service.one(jobs[0]["context"])
    assert len(requests) == 3 and cancelled == [{"terminate_containers": False}] * 3


def test_media_retention_deduplicates_without_changing_wire_payload(tmp_path):
    _, jobs = fixture()
    payload = jobs[1]["context"].request(TextTokenizer(), defaults(), 2048)
    before = copy.deepcopy(payload)
    retained = retain_payload(payload, tmp_path)
    assert retained == retain_payload(payload, tmp_path)
    assert payload == before and len(list((tmp_path / "images").iterdir())) == 1
    replay = copy.deepcopy(retained)
    for part in replay["contents"][0]["parts"]:
        if "retainedImage" in part:
            ref = part.pop("retainedImage")
            body = Path(ref["path"]).read_bytes()
            assert sha(body) == ref["sha256"] and len(body) == ref["bytes"]
            part["inlineData"] = {"mimeType": ref["mimeType"], "data": base64.b64encode(body).decode()}
    assert canonical(replay) == canonical(payload)


def journal_fixture(tmp_path):
    from opencode_rl.store import TraceStore, atomic_json
    data, jobs = fixture()
    row = data.metadata["ocrl_rows"][0]
    row.update(trajectory_id="b"*32 + "-r000", contract_sha256="contract")
    root = tmp_path / "groups" / ("b"*32) / "r000"
    attempt = root / "attempt_000"
    aid = "a"*32
    atomic_json(attempt / "attempt.json", {"attempt_id": aid})
    atomic_json(root / "admitted.json", {"contract_sha256": "contract", "attempt": {
        "trajectory_id": row["trajectory_id"], "task_id": row["task_id"], "attempt_id": aid,
        "runtime_completed": True, "infrastructure_error": False,
        "request_ids": [s["request_id"] for s in row["causal_spans"]]}})
    store = TraceStore(attempt / "trace")
    runtime = store.new_request({"kind": "append_task_runtime", "attempt_id": aid}, {})
    loop = store.new_request({"kind": "append_token_loop", "attempt_id": aid}, {})
    runtime.event("tool_host_frame", {"kind": "profile", "blob": store.blob(canonical({"profile": {
        "task": "original task", "tools": TOOLS}}))})
    runtime.seal("complete")
    for job in jobs[:2]:
        c = job["context"]
        loop.event("generation_requested", {"metadata": {"turn": c.span["native_turn"], "attempt_id": aid},
            "prompt_token_ids": c.prompt_ids(), "features": c.features})
        loop.event("generation_complete", {"turn": {"request_id": c.span["request_id"], "input": {
            "input_sha256": c.span["input_sha256"], "features": c.features}}})
        loop.event("tool_step_complete", {"results": [{"attachments": [{"url": URL}]}]})
    loop.seal("complete")
    return data, jobs, row, loop.path


def test_actual_sealed_journal_reconstructs_history_and_image_prefix(tmp_path):
    data, jobs, row, path = journal_fixture(tmp_path)
    contexts = contexts_for_row({"run_root": str(tmp_path)}, 0, row, jobs[0]["context"].sequence)
    assert [c.prompt_ids() for c in contexts] == [j["context"].prompt_ids() for j in jobs[:2]]
    assert contexts[0].image_urls == () and contexts[1].image_urls == (URL,)
    assert contexts[1].tools == TOOLS and len(contexts) == 2
    # The second action's following image is not in the second prompt.
    assert len(contexts[1].features) == 1
    lines = (path / "events.jsonl").read_text().splitlines()
    lines[1] = lines[1].replace('"turn":0', '"turn":7')
    (path / "events.jsonl").write_text("\n".join(lines)+"\n")
    with pytest.raises(ValueError, match="seal"):
        list(checked_events(path))


def test_base_teacher_cannot_be_updated_and_resume_does_not_load_ref():
    with pytest.raises(ValueError, match="permanently fixed"):
        AnswerOpsdTrainerMixin.__new__(AnswerOpsdTrainerMixin).update_ref_with_policy()
    root = source_root()
    tree = ast.parse((root / "skyrl/train/trainer.py").read_text())
    load = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_checkpoints")
    code = ast.unparse(load)
    assert "'policy'" in code and "'critic'" in code and "'ref'" not in code


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_new_mode_only_changes_privilege_and_binds_resume(tmp_path, direction):
    source = {"harness": "append-opencode-tools-v1", "causal_prefix": True, "run_root": "/old/run",
        "checkpoint_root": "/old/ckpt", "model_path": "/model/pinned-base", "contract": {"max_context": 131072},
        "engines": [{"trace_root": "/old/run/engine0/trace"}], "prepared_sha256": "train", "grader_hashes": {}}
    options = {"run_root": str(tmp_path/"new"), "checkpoint_root": str(tmp_path/"ckpt"),
               "runtime_profile": "source", "direction": direction}
    old = new_plan(source, **options, private_answers_path="/private/answers.json", private_answers_sha256="sha")
    new = new_plan(source, **options, privileged_information=MODE)
    for key in new:
        if key != "opsd":
            assert old[key] == new[key]
    extra = {"private_answers_path", "private_answers_sha256", "privileged_information", "hint_placement",
             "demonstrator", "demonstration_admission"}
    assert {k: v for k, v in new["opsd"].items() if k not in extra} == {k: v for k, v in old["opsd"].items() if k not in extra}
    assert new["opsd"]["teacher_model_path"] == source["model_path"] and new["opsd"]["teacher_mode"] == "fixed_initial"
    assert new["opsd"]["hint_placement"] == "after_original_turn_context"
    assert new["opsd"]["demonstration_admission"] == "whole-trajectory-demo-availability-v2"
    receipt = tmp_path/"resume"
    receipt.mkdir()
    (receipt/"opsd_resume.json").write_bytes(canonical({"step": 1, "identity": objective_identity(new)}))
    check_resume(receipt, new)
    for other in (old, copy.deepcopy(new)):
        if other is not old:
            other["opsd"]["hint_placement"] = "system"
        with pytest.raises(ValueError, match="changed"):
            check_resume(receipt, other)
    with pytest.raises(ValueError, match="private answers"):
        new_plan(source, **options, privileged_information=MODE, private_answers_path="/answers")
