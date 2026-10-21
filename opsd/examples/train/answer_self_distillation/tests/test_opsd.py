from examples.train.answer_self_distillation.tests.source_fixtures import source_root
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from examples.train.answer_self_distillation.answers import (
    PrivateAnswers, answer_card, build_catalog, canonical, sha,
)
from examples.train.answer_self_distillation.configuration import check_objective
from examples.train.answer_self_distillation.launch import check_resume, new_plan, objective_identity, training_command
from examples.train.answer_self_distillation.objective import per_turn_audit, sampled_advantages
from examples.train.answer_self_distillation.teacher_inputs import encode_hint, make_teacher_batch
from examples.train.answer_self_distillation.trainer import AnswerOpsdTrainerMixin
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch, TensorList


class Tokenizer:
    all_special_ids = [0, 1, 2]
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>system\n":
            return [1, 3, 4]
        return [10 + ord(c) for c in text]

    def convert_tokens_to_ids(self, value):
        assert value == "<|im_end|>"
        return 2


def batch(n=2):
    # Prompt=[system, public instructions, end, user]; response spans include a
    # tool/image observation gap at [2:4]. The second row is shorter/left padded.
    sequences = torch.tensor([[1, 3, 4, 20, 2, 21, 30, 31, 99, 98, 32, 33],
                              [0, 0, 1, 3, 4, 22, 2, 23, 34, 35, 36, 37]])
    mask = torch.tensor([[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1]])
    types = torch.zeros_like(sequences)
    types[0, 8:10] = 1
    data = TrainingInputBatch({"sequences": sequences, "attention_mask": (sequences != 0).long(),
        "loss_mask": mask, "response_mask": mask.clone(), "mm_token_type_ids": types,
        "ocrl_packet_length": torch.tensor([[6], [4]]),
        "pixel_values": TensorList([torch.arange(6).view(2, 3).float(), torch.empty(0, 3)]),
        "image_grid_thw": TensorList([torch.tensor([[1, 1, 2]]), torch.empty(0, 3, dtype=torch.long)])})
    data.metadata = {"response_length": 6, "ocrl_rows": [
        {"task_id": "t1", "trajectory_id": "traj1", "outcome": 0., "causal_spans": [
            {"request_id": "t1-r1", "native_turn": 1, "response_start": 0, "response_end": 2},
            {"request_id": "t1-r2", "native_turn": 2, "response_start": 4, "response_end": 6}]},
        {"task_id": "t2", "trajectory_id": "traj2", "outcome": 1., "causal_spans": [
            {"request_id": "t2-r1", "native_turn": 1, "response_start": 0, "response_end": 4}]}]}
    return data


def test_teacher_only_insertion_exact_suffix_media_masks_and_no_mutation():
    data = batch()
    before = {k: v.clone() for k, v in data.items() if isinstance(v, torch.Tensor)}
    teacher, metrics = make_teacher_batch(data, [[60, 61], [70, 71, 72]], Tokenizer(), 30)
    assert teacher is not data
    for key, value in before.items():
        assert torch.equal(data[key], value), key
    for i in range(2):
        live = data["loss_mask"][i].bool()
        assert torch.equal(data["sequences"][i, -6:][live], teacher["sequences"][i, -6:][live])
    assert teacher["pixel_values"] is data["pixel_values"]
    assert teacher["image_grid_thw"] is data["image_grid_thw"]
    assert teacher["response_mask"] is data["response_mask"]
    assert teacher["mm_token_type_ids"].sum() == data["mm_token_type_ids"].sum() == 2
    assert teacher["sequences"][0][teacher["attention_mask"][0].bool()].tolist()[:8] == [1, 3, 4, 20, 60, 61, 2, 21]
    assert metrics["opsd/hint_tokens_mean"] == 2.5
    assert "ocrl_rows" not in teacher.metadata and "loss_mask" not in teacher


@pytest.mark.parametrize("case", ["context", "header", "right_padding", "inside_action", "hint_count"])
def test_teacher_fails_closed(case):
    data, hints, cap = batch(), [[60], [70]], 30
    if case == "context":
        cap = 12
    elif case == "header":
        data["sequences"][0, 0] = 11
    elif case == "right_padding":
        data["attention_mask"][0, -1] = 0
    elif case == "inside_action":
        data["ocrl_packet_length"][0, 0] = 9
    else:
        hints.pop()
    with pytest.raises(ValueError):
        make_teacher_batch(data, hints, Tokenizer(), cap)


def test_hint_budget_and_special_token_rejection():
    with pytest.raises(ValueError, match="budget"):
        encode_hint(Tokenizer(), "abcd", 3)
    tok = Tokenizer()
    tok.all_special_ids = [0, 1, 2, 107]
    with pytest.raises(ValueError, match="special"):
        encode_hint(tok, "a", 9)


def test_objective_k1_sign_clip_mask_stopgrad_and_nonfinite():
    student = torch.tensor([[-2., -1., float("nan"), -10.]], requires_grad=True)
    teacher = torch.tensor([[-1., -3., float("inf"), -1.]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 1]])
    advantage, metrics = sampled_advantages(teacher, student, mask, 5)
    assert advantage.tolist() == [[1., -2., 0., 5.]]
    assert not advantage.requires_grad and metrics["opsd/gap_clip_fraction"] == pytest.approx(1/3)
    assert metrics["opsd/outcome_loss_weight"] == 0
    with pytest.raises(ValueError, match="Nonfinite"):
        sampled_advantages(teacher, student, torch.ones_like(mask))
    with pytest.raises(ValueError, match="No student"):
        sampled_advantages(teacher, student, torch.zeros_like(mask))


def test_actual_skyrl_rollout_is_gradient_sign_and_behavior_guard():
    from skyrl.train.config import AlgorithmConfig
    from skyrl.backends.skyrl_train.utils.ppo_utils import rollout_is_policy_loss
    config = AlgorithmConfig()
    config.eps_clip_low, config.eps_clip_high = .2, .28
    current = torch.tensor([[-2., -2., -2., -2.]], requires_grad=True)
    behavior = torch.tensor([[-2., -2., -2., -5.]])
    teacher = torch.tensor([[-1., -3., -1., -1.]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 1]])
    advantage, _ = sampled_advantages(teacher, current, mask)
    loss, metrics = rollout_is_policy_loss(current, current.detach(), advantage, config,
                                           loss_mask=mask, rollout_logprobs=behavior)
    loss.sum().backward()
    assert current.grad[0, 0] < 0  # descent raises this token's probability
    assert current.grad[0, 1] > 0  # descent lowers this token's probability
    assert current.grad[0, 2] == current.grad[0, 3] == 0  # tool and too-off-policy
    assert teacher.grad is None
    assert metrics["clip_ratio"] > 0


def test_turn_mapping_uses_native_spans_and_request_ids_not_reward():
    data = batch()
    teacher, student = torch.full((2, 6), -1.), torch.full((2, 6), -2.)
    advantage, _ = sampled_advantages(teacher, student, data["loss_mask"])
    answers = NS(get=lambda task: {"stage": "s_1", "answer_sha256": "digest-"+task})
    rows = per_turn_audit(data, teacher, student, advantage, answers)
    assert [t["request_id"] for t in rows[0]["turns"]] == ["t1-r1", "t1-r2"]
    assert [t["trajectory_guidance_fraction"] for t in rows[0]["turns"]] == [.5, .5]
    assert rows[0]["guidance_abs_sum"] == rows[1]["guidance_abs_sum"] == 4
    assert rows[0]["outcome_reward"] == 0 and rows[0]["outcome_loss_weight"] == 0


def test_zero_outcome_group_survives_stale_group_does_not():
    trainer = AnswerOpsdTrainerMixin.__new__(AnswerOpsdTrainerMixin)
    trainer.opsd_plan = {"opsd": {"privileged_information": "answers"}}
    trainer._check_ocrl_configuration = lambda: None
    trainer.global_step, trainer.max_staleness_steps = 3, 1
    trainer.private_answers = NS(get=lambda key: {})
    group = NS(uid="x", global_step_when_scheduled=2, generator_output={"rewards": [0.],
        "loss_masks": [[1, 0, 1]], "ocrl_rows": [{"task_id": "x", "causal_spans": [1]}]})
    assert trainer._should_keep_group(group)
    group.global_step_when_scheduled = 1
    assert not trainer._should_keep_group(group)
    group.global_step_when_scheduled = 4
    with pytest.raises(ValueError, match="Future"):
        trainer._should_keep_group(group)


def test_rewards_replaced_not_combined():
    trainer = AnswerOpsdTrainerMixin.__new__(AnswerOpsdTrainerMixin)
    trainer.opsd_plan, trainer.all_metrics = {"opsd": {"gap_clip": 5}}, {}
    a = batch()
    a["base_action_log_probs"], a["action_log_probs"] = torch.full((2, 6), -1.), torch.full((2, 6), -2.)
    for outcome in (-999., 0., 1., 999.):
        a["rewards"] = torch.full((2, 6), outcome)
        trainer.apply_reward_kl_penalty(a)
        assert torch.equal(a["rewards"], a["loss_mask"].float())


def test_s4e_partial_answer_handles_missing_units_dips_and_preroll():
    manifest = {"taskId": "s4e", "stretch_dur_s": 4., "shots": [{"id": "p01", "unit": "s1"}, {"id": "p03", "unit": "s3"}],
                "joins": [{"join": 1, "leaving": "p01", "arriving": "p03"}], "dialogue": [{"id": "d02"}]}
    answer = {"task": "s4e", "n_joins": 1, "golden_joins": [{"cut": 1, "picture": "dip_to_black", "black_hold_s": .4}],
              "golden_dialogue": [{"id": "d02", "text": "hello", "trailer_s": -.2}], "music_align_s": 1.5}
    result = answer_card("s_4e", answer, manifest)
    assert result["reference_joins"][0]["before_clip"] == "p03"
    assert result["reference_joins"][0]["black_hold_s"] == .4
    assert result["reference_dialogue"][0]["preview_onset_s"] == -.2
    assert result["reference_is_not_unique"] and "NOT" in result["reference_kind"]
    assert "dialogue take trim endpoints" in result["unknown_not_ground_truth"]
    assert not any(k in result["reference_dialogue"][0] for k in ("trim_in", "trim_out", "gain"))


def test_s2_keeps_alternative_answers():
    result = answer_card("s_2", {"task": "s2", "order": ["a", "b"], "alternative_orders": [["b", "a"]]},
                        {"taskId": "s2", "clips": [{"id": "a"}, {"id": "b"}]})
    assert result["equally_accepted_alternative_orders"] == [["b", "a"]]


def test_same_actual_loop_and_methods_no_private_generator_import():
    from examples.train.answer_self_distillation.generator import AnswerOpsdGenerator
    from opencode_rl.train_generator import OpenCodeGenerator
    from append_agent_rl.generator import AppendGenerator
    import examples.train.answer_self_distillation.generator as module
    for method in ("generate", "_one", "_completed_rollout", "evaluation_one"):
        assert getattr(AnswerOpsdGenerator, method) is getattr(AppendGenerator, method) is getattr(OpenCodeGenerator, method)
    assert "private_answers" not in Path(module.__file__).read_text()


def test_opt_in_core_filter_keeps_old_default_and_staleness():
    from skyrl.train.config import FullyAsyncConfig
    assert FullyAsyncConfig().group_filter == "reward_variance"
    # Execute the actual method body without importing GPU-only trainer dependencies.
    source = source_root() / "skyrl/train/fully_async_trainer.py"
    tree = ast.parse(source.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "FullyAsyncRayPPOTrainer")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_should_keep_group")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    calls = []
    scope = {"zero_variance_filter": lambda *a, **k: calls.append(a) or [0]}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), scope)
    trainer = NS(cfg=NS(trainer=NS(fully_async=NS(group_filter="validity"), algorithm=NS(zero_variance_filter_tol=1e-6))),
                 global_step=3, max_staleness_steps=1)
    group = NS(uid="a", global_step_when_scheduled=2, generator_output={"rewards": [0.], "loss_masks": [[1]]})
    assert scope["_should_keep_group"](trainer, group) and not calls
    trainer.cfg.trainer.fully_async.group_filter = "reward_variance"
    assert scope["_should_keep_group"](trainer, group) and len(calls) == 1


def test_registry_is_safe_to_initialize_twice():
    from examples.train.answer_self_distillation.main_answer_opsd import register_objective
    register_objective()
    register_objective()


def test_teacher_packing_equals_separate_turn_guidance_and_gradients():
    torch.manual_seed(11)

    class TinyCausal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.pos = torch.nn.Embedding(128, 8), torch.nn.Embedding(32, 8)
            self.attention = torch.nn.MultiheadAttention(8, 2, dropout=0, batch_first=True)
            self.head = torch.nn.Linear(8, 128)

        def forward(self, seq):
            x = self.emb(seq)[None] + self.pos(torch.arange(len(seq)))[None]
            mask = torch.ones(len(seq), len(seq), dtype=torch.bool).triu(1)
            x, _ = self.attention(x, x, x, attn_mask=mask)
            return self.head(x[0]).log_softmax(-1)[:-1].gather(-1, seq[1:, None]).flatten()

    packed_model = TinyCausal()
    separate_model, teacher = copy.deepcopy(packed_model), copy.deepcopy(packed_model)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    seq = batch()["sequences"][0]
    teacher_seq = torch.cat((seq[:4], torch.tensor([60, 61]), seq[4:]))
    student_lp, teacher_lp = packed_model(seq)[-6:], teacher(teacher_seq)[-6:]
    mask = torch.tensor([1, 1, 0, 0, 1, 1])
    advantage, _ = sampled_advantages(teacher_lp, student_lp, mask)
    packed_loss = -(advantage * student_lp).sum()/4
    packed_loss.backward()
    separate_loss = 0.
    for lo, hi in ((6, 8), (10, 12)):
        student_turn = separate_model(seq[:hi])[lo-1:hi-1]
        teacher_turn = teacher(teacher_seq[:hi+2])[lo+1:hi+1]
        turn_adv, _ = sampled_advantages(teacher_turn, student_turn, torch.ones(hi-lo))
        assert torch.allclose(turn_adv, advantage[lo-6:hi-6], atol=1e-6)
        separate_loss = separate_loss - (turn_adv * student_turn).sum()/4
    separate_loss.backward()
    assert torch.allclose(packed_loss, separate_loss, atol=1e-6)
    for left, right in zip(packed_model.parameters(), separate_model.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=1e-6)
    assert all(p.grad is None for p in teacher.parameters())


@pytest.mark.parametrize("direction", ["reverse", "forward"])
def test_launch_preserves_task_batch_optimizer_and_async_limits(tmp_path, direction):
    from opencode_rl.train_config import config_class
    source = {"harness": "append-opencode-tools-v1", "causal_prefix": True, "run_root": "/old/run",
        "checkpoint_root": "/old/checkpoint", "mode": "train", "model_path": "/model/pinned-base",
        "contract": {"max_context": 131072, "max_output": 16384, "temperature": 1., "top_p": 1., "top_k": -1},
        "engines": [{"url": f"http://localhost:{18000+i}", "trace_root": f"/old/run/engine_{i}/trace"} for i in range(4)],
        "eval_interval": 10, "validation": {"manifest": "same-val"}, "prepared_sha256": "train-hash",
        "grader_hashes": {"grader": "unchanged"}, "resume": {"step": 30}, "direct_start_authorization": {"old": True}}
    baseline = copy.deepcopy(source)
    plan = new_plan(source, run_root=str(tmp_path/"newrun"), checkpoint_root=str(tmp_path/"newckpt"),
                    private_answers_path="/private/answers.json", private_answers_sha256="a"*64, runtime_profile="source", direction=direction)
    assert plan["opsd"]["loss"] == f"teacher_topk_tail_{direction}_kl"
    assert plan["opsd"]["kl_direction"] == direction
    assert plan["opsd"]["top_k"] == 64 and plan["opsd"]["topk_tail"] is True
    assert plan["opsd"]["gap_clip"] is None and plan["opsd"]["action_importance_weighting"] is False
    base_argv = ["python", "-m", "old_entrypoint", "trainer.train_batch_size=16", "trainer.policy_mini_batch_size=16",
        "generator.n_samples_per_prompt=8", "trainer.policy.optimizer_config.lr=0.000001",
        "trainer.max_tokens_per_microbatch=65536", "trainer.fully_async.max_staleness_steps=1",
        "trainer.fully_async.num_parallel_generation_workers=32", "trainer.algorithm.eps_clip_low=0.2",
        "trainer.algorithm.eps_clip_high=0.28"]
    argv = training_command(base_argv, "/plans/newplan.json", plan)
    values = dict(s.split("=", 1) for s in argv[3:])
    for key in ("trainer.train_batch_size", "trainer.policy_mini_batch_size", "trainer.policy.optimizer_config.lr",
                "trainer.max_tokens_per_microbatch", "trainer.fully_async.max_staleness_steps",
                "trainer.fully_async.num_parallel_generation_workers", "trainer.algorithm.eps_clip_low", "trainer.algorithm.eps_clip_high"):
        assert values[key] == dict(s.split("=", 1) for s in base_argv[3:])[key]
    cfg = config_class().from_cli_overrides(argv[3:])
    check_objective(cfg)
    assert cfg.generator.n_samples_per_prompt == 1
    assert cfg.trainer.algorithm.policy_loss_type == "answer_opsd_topk"
    assert source == baseline and plan["validation"] == source["validation"]
    assert "resume" not in plan and "direct_start_authorization" not in plan
    with pytest.raises(ValueError, match="old GRPO"):
        check_resume(tmp_path/"old-grpo", plan)
    receipt = tmp_path/"resume"
    receipt.mkdir()
    (receipt/"opsd_resume.json").write_bytes(canonical({"step": 1, "identity": objective_identity(plan)}))
    check_resume(receipt, plan)
    other = copy.deepcopy(plan)
    other["opsd"]["kl_direction"] = "reverse" if direction == "forward" else "forward"
    other["opsd"]["loss"] = f"teacher_topk_tail_{other['opsd']['kl_direction']}_kl"
    with pytest.raises(ValueError, match="same-identity resume only"):
        check_resume(receipt, other)
    changed = copy.deepcopy(plan)
    changed["opsd"]["gap_clip"] = 2.
    with pytest.raises(ValueError, match="same-identity resume only"):
        check_resume(receipt, changed)
    cfg.trainer.algorithm.zero_variance_filter = True
    with pytest.raises(ValueError, match="normalization"):
        check_objective(cfg)


def test_production_multimodal_patch_anchors_still_apply():
    from opencode_rl.install_training_fixes import transform
    root = source_root()
    for kind, path in (("fully_async", "skyrl/train/fully_async_trainer.py"), ("trainer", "skyrl/train/trainer.py")):
        patched = transform(kind, (root/path).read_text())
        compile(patched, path, "exec")
        if kind == "fully_async":
            assert 'group_filter == "validity"' in patched


def test_private_catalog_integrity_split_and_source_binding(tmp_path):
    root = tmp_path/"source"
    (root/"verifier").mkdir(parents=True)
    (root/"initial_state").mkdir()
    a, m = {"task": "a", "answer": ["c01"], "n_answer": 1}, {"taskId": "a", "candidates": [{"id": "c01"}]}
    names = {"verifier/answer.json": canonical(a), "initial_state/manifest.json": canonical(m)}
    for name, content in names.items():
        (root/name).write_bytes(content)
    row = {"task": "train1", "stage": "s_1", "task_path": str(root), "probe": {"split": "train"},
           "files": {name: {"sha256": sha(content)} for name, content in names.items()}}
    val = {**row, "task": "val1", "probe": {"split": "val"}}
    release = tmp_path/"release.json"
    release.write_bytes(canonical({"tasks": [row, val], "train": 1}))
    catalog = build_catalog(release)
    path = tmp_path/"answers.json"
    path.write_bytes(canonical(catalog))
    digest = sha(path.read_bytes())
    prepared = {"train1": {"source_hashes": {k: sha(v) for k, v in names.items()}}}
    answers = PrivateAnswers(path, digest, release, prepared)
    assert set(answers.tasks) == {"train1"}
    with pytest.raises(ValueError, match="validation"):
        answers.get("val1")
    with pytest.raises(ValueError, match="digest"):
        PrivateAnswers(path, "0"*64, release)
    prepared["train1"]["source_hashes"]["verifier/answer.json"] = "bad"
    with pytest.raises(ValueError, match="prepared"):
        PrivateAnswers(path, digest, release, prepared)
    (root/"verifier/answer.json").write_text("{}")
    with pytest.raises(ValueError, match="Released"):
        build_catalog(release)


def test_forward_hooks_never_send_hint_to_student_and_keep_dp_row_identity():
    data = batch()
    trainer = AnswerOpsdTrainerMixin.__new__(AnswerOpsdTrainerMixin)
    trainer.turn_demonstrations = None
    trainer._check_ocrl_configuration = lambda: None
    trainer.has_critic, trainer.ref_model = False, object()
    trainer.tokenizer, trainer.all_metrics = Tokenizer(), {}
    trainer.opsd_plan = {"opsd": {"teacher_max_context": 32}}
    trainer._hint_ids = {"t1": [60], "t2": [70, 71]}
    trainer.dispatch = NS(empty_cache=lambda *a: None)
    recorded = {}
    def forward(model, batch, **kwargs):
        recorded[model] = batch
        return torch.full((2, 6), -1. if model == "ref" else -2.)
    trainer._execute_forward_pass = forward
    data["rollout_logprobs"] = torch.full((2, 6), -2.)
    result = trainer.fwd_logprobs_values_reward(data)
    assert result is data
    assert recorded["policy"]["sequences"] is data["sequences"]
    assert 60 not in recorded["policy"]["sequences"].flatten().tolist()
    assert 60 in recorded["ref"]["sequences"][0].tolist()
    assert 70 in recorded["ref"]["sequences"][1].tolist()
    assert result["values"] is None and torch.equal(result["base_action_log_probs"], torch.full((2, 6), -1.))


def test_installer_preserves_frozen_safety_patches_and_matches_core_ast():
    import subprocess
    from examples.train.answer_self_distillation.install import extend_source, PATHS
    from opencode_rl.install_training_fixes import transform
    root = source_root()
    for kind, relative in PATHS.items():
        old = subprocess.check_output(["git", "show", "HEAD:"+relative], cwd=root, text=True)
        if kind == "fully_async":
            extended = extend_source(kind, old)
            assert ast.dump(ast.parse(extended)) == ast.dump(ast.parse((root/relative).read_text()))
            frozen = transform("fully_async", old) + "\n# S12_ASYNC_FINAL_CHECKPOINT_STEP\n# S12_ASYNC_MAX_STEP_RANGE\n"
            extended = extend_source(kind, frozen)
            assert "S12_ASYNC_FINAL_CHECKPOINT_STEP" in extended and "OCRL_EXACT_MEDIA_V7" in extended
            compile(extended, relative, "exec")
        else:
            extended = extend_source(kind, old)
            assert 'group_filter: Literal["reward_variance", "validity"] = "reward_variance"' in extended
        with pytest.raises(ValueError, match="Already"):
            extend_source(kind, extended)


def test_actual_k1_constructor_does_not_weaken_grpo_guard(tmp_path):
    from append_agent_rl.tests.test_backend import contract
    from append_agent_rl.generator import AppendGenerator
    from append_agent_rl.task_rollout import rollout_task
    from examples.train.answer_self_distillation.generator import AnswerOpsdGenerator
    c, config = contract()
    kwargs = dict(dataset=object(), contract=c, config=config, app=object(), image_id="test-image",
                  engines=[{"url": "http://127.0.0.1:18081"}], root=tmp_path, grader=object(), commit=object())
    assert AnswerOpsdGenerator(**kwargs).k == 1
    assert AnswerOpsdGenerator(**kwargs).rollout is rollout_task
    with pytest.raises(ValueError):
        AppendGenerator(**kwargs, k=1)
    assert AppendGenerator(**kwargs, k=8).k == 8
    with pytest.raises(ValueError):
        AnswerOpsdGenerator(**kwargs, k=8)


def test_frozen_profile_reuses_existing_methods_without_fake_resume(monkeypatch, tmp_path):
    directory = os.environ.get("OPSD_TEST_FROZEN_ADAPTER_DIR")
    if not directory:
        pytest.skip("Set OPSD_TEST_FROZEN_ADAPTER_DIR for actual deployed adapter method checks")
    monkeypatch.syspath_prepend(directory)
    from append_resume_guard import adapter_hash
    from append_resume_preparation import PreparedAppendGenerator
    from append_resume_delivery import rollout_task
    from append_resume_checkpoint import BackgroundCheckpointPersistence
    from examples.train.answer_self_distillation.frozen_runtime import (
        EXTENSIONS, generator_class, persistence_class, verify_frozen_extensions,
    )
    from append_agent_rl.tests.test_backend import contract
    cls = generator_class()
    c, config = contract()
    generator = cls(dataset=object(), contract=c, config=config, app=object(), image_id="test-image",
                    engines=[{"url": "http://127.0.0.1:18081"}], root=tmp_path, grader=object(), commit=object())
    assert generator.k == 1 and generator.rollout is rollout_task
    for name in ("_completed_rollout", "_one", "generate", "evaluation_one"):
        assert getattr(cls, name) is getattr(PreparedAppendGenerator, name)
    plan = {"run_root": str(tmp_path), "opsd": {"runtime_extensions": {**EXTENSIONS, "adapter_sha256": adapter_hash()}}}
    assert "resume" not in plan
    verify_frozen_extensions(plan)
    persistence_type = persistence_class()
    for name in ("save", "_publish", "wait_for_publication", "close"):
        assert getattr(persistence_type, name) is getattr(BackgroundCheckpointPersistence, name)
    persistence = persistence_type(plan, NS(commit=lambda: None), NS(commit=lambda: None))
    persistence.close()
    plan["opsd"]["runtime_extensions"]["adapter_sha256"] = "bad"
    with pytest.raises(ValueError, match="differs"):
        verify_frozen_extensions(plan)
