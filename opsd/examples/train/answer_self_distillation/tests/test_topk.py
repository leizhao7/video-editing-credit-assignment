from examples.train.answer_self_distillation.tests.source_fixtures import source_file, source_root
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from examples.train.answer_self_distillation.topk import (
    OBJECTIVE, LOSS_NAME, bucket_logprobs, categorical_kl, distribution_kl,
    teacher_targets, project_response, selected_token_fallback_forbidden,
)
from examples.train.answer_self_distillation.topk_trainer import expand_live, per_turn_audit
from examples.train.answer_self_distillation.topk_install import extend_topk_source
from examples.train.answer_self_distillation.trainer import AnswerOpsdTrainerMixin
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch, TensorList, pad_training_input_batch


def config(k=3):
    return {"loss": OBJECTIVE, "gap_clip": None, "top_k": k, "topk_tail": True,
            "projection_chunk_size": 2, "action_importance_weighting": False}


@pytest.mark.parametrize("direction", ["reverse", "forward"])
@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_chunked_loss_and_gradients_equal_dense(direction, chunk):
    torch.manual_seed(71)
    h = torch.randn(7, 5, dtype=torch.double, requires_grad=True)
    w = torch.randn(13, 5, dtype=torch.double, requires_grad=True)
    actions = torch.arange(7)
    ids, teacher, _ = teacher_targets(h.detach() + .2, w.detach() * .6, actions, 4, 3)
    expected_h, expected_w = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    expected = categorical_kl(bucket_logprobs(expected_h @ expected_w.T, ids), teacher, direction)
    actual, lp, entropy, mass = distribution_kl(h, w, ids, teacher, actions, chunk, direction)
    weights = torch.arange(1, 8, dtype=torch.double) / 28
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    assert torch.allclose(actual, expected, atol=1e-11)
    assert torch.allclose(h.grad, expected_h.grad, atol=1e-10)
    assert torch.allclose(w.grad, expected_w.grad, atol=1e-10)
    assert not any(t.requires_grad for t in (teacher, lp, entropy, mass))


def test_vocab_minus_one_topk_equals_full_vocab_kl_and_gradient():
    torch.manual_seed(12)
    h = torch.randn(2, 4, dtype=torch.double)
    w = torch.randn(7, 4, dtype=torch.double, requires_grad=True)
    teacher_z = torch.randn(2, 7, dtype=torch.double)
    ids = teacher_z.topk(6, -1).indices
    target = bucket_logprobs(teacher_z, ids)
    actual = distribution_kl(h, w, ids, target, torch.tensor([1, 2]))[0].sum()
    full = categorical_kl((h @ w.T).log_softmax(-1), teacher_z.log_softmax(-1)).sum()
    assert torch.allclose(actual, full, atol=1e-12)
    a, = torch.autograd.grad(actual, w, retain_graph=True)
    b, = torch.autograd.grad(full, w)
    assert torch.allclose(a, b, atol=1e-10)


def test_unsampled_teacher_token_gets_positive_training_signal():
    # Student sampled token 0, but the teacher prefers unobserved candidate 1.
    h = torch.ones(1, 1)
    w = torch.tensor([[1.], [-1.], [0.], [-2.]], requires_grad=True)
    teacher_w = torch.tensor([[-1.], [3.], [0.], [-2.]], requires_grad=True)
    ids, target, _ = teacher_targets(h, teacher_w, torch.tensor([0]), 2, 1)
    loss = distribution_kl(h, w, ids, target, torch.tensor([0]))[0].sum()
    loss.backward()
    assert 1 in ids and w.grad[1] < 0  # gradient descent raises unsampled token 1
    assert teacher_w.grad is None
    # Distribution KL does not depend on WHICH action was sampled at this prefix.
    other = distribution_kl(h, w, ids, target, torch.tensor([3]))[0]
    assert torch.equal(loss.detach(), other.sum())


def test_tail_keeps_full_normalizer_and_is_stable_for_extreme_logits():
    z = torch.tensor([[1000., 999., -1000., -1001.]])
    ids = torch.tensor([[0, 1]])
    bins = bucket_logprobs(z, ids)
    assert torch.isfinite(bins).all() and bins[0, -1] < -1900
    assert torch.allclose(bins.logsumexp(-1), torch.zeros(1), atol=1e-5)
    other = bucket_logprobs(torch.tensor([[.1, .2, .3, .4]]), ids)
    assert other[0, :2].exp().sum() < 1  # never renormalize top-K to one
    for direction in ("reverse", "forward"):
        w = z.T.clone().requires_grad_()
        loss = distribution_kl(torch.ones(1, 1), w, ids, other, torch.tensor([0]), direction=direction)[0].sum()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(w.grad).all()


def test_zero_live_padding_executes_zero_backward():
    h = torch.randn(2, 8, 3, requires_grad=True)
    w = torch.randn(9, 3, requires_grad=True)
    req = {"mode": "train", "mask": torch.zeros(2, 5, dtype=torch.bool), "chunk_size": 2,
           "direction": "reverse", "ids": torch.empty(0, 3, dtype=torch.long), "teacher": torch.empty(0, 4)}
    result = project_response(h, w, torch.zeros(2, 8, dtype=torch.long), 5, req)
    result["kl"].sum().backward()
    assert not h.grad.any() and not w.grad.any()


def test_alignment_predicts_t_plus_one_and_excludes_context_gap():
    torch.manual_seed(4)
    h = torch.randn(1, 8, 4, requires_grad=True)
    w = torch.randn(11, 4, requires_grad=True)
    seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    mask = torch.tensor([[1, 0, 0, 1, 1]], dtype=torch.bool)
    req = {"mode": "teacher", "mask": mask, "k": 3, "chunk_size": 2}
    target = project_response(h.detach() + .1, w.detach(), seq, 5, req)
    req.update(mode="train", ids=target["ids"], teacher=target["teacher"], direction="reverse")
    result = project_response(h, w, seq, 5, req)
    result["kl"].sum().backward()
    selected = torch.tensor([2, 5, 6])  # predicts response positions 3,6,7
    actual = (h.grad.abs().sum(-1) > 0).nonzero()[:, 1]
    assert torch.equal(actual, selected)
    dense = (h.detach()[0, selected] @ w.detach().T).log_softmax(-1)
    assert torch.allclose(result["logprobs"], dense.gather(-1, seq[0, selected+1, None]).flatten(), atol=1e-6)


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)
        self.attention = torch.nn.MultiheadAttention(8, 2, dropout=0., batch_first=True)
        self.weight = torch.nn.Parameter(torch.randn(128, 8) / 4)

    def forward(self, seq, n, attention_mask=None, **kwargs):
        x = self.embedding(seq)
        # A real causal attention backbone: future tool/hint tokens cannot leak.
        causal = torch.ones(seq.shape[1], seq.shape[1], dtype=torch.bool).triu(1)
        x, _ = self.attention(x, x, x, attn_mask=causal)
        return project_response(x, self.weight, seq, n, kwargs["opsd_request"])


def test_packed_distribution_loss_equals_per_turn_gradients():
    torch.manual_seed(27)
    actor = TinyActor()
    other, teacher = copy.deepcopy(actor), copy.deepcopy(actor)
    seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    teacher_seq = torch.cat([seq[:, :2], torch.tensor([[100, 101]]), seq[:, 2:]], -1)
    mask = torch.tensor([[1, 1, 0, 0, 1, 1]], dtype=torch.bool)
    req = {"mode": "teacher", "mask": mask, "k": 4, "chunk_size": 2}
    with torch.no_grad():
        targets = teacher(teacher_seq, 6, opsd_request=req)
    req.update(mode="train", direction="reverse", ids=targets["ids"], teacher=targets["teacher"])
    loss = actor(seq, 6, opsd_request=req)["kl"].sum() / 4
    loss.backward()
    separate_loss = 0.
    for index, (lo, hi) in enumerate(((3, 5), (7, 9))):
        part = {"mode": "teacher", "mask": torch.ones(1, 2, dtype=torch.bool), "k": 4, "chunk_size": 1}
        with torch.no_grad():
            t = teacher(teacher_seq[:, :hi+2], 2, opsd_request=part)
        assert torch.equal(t["ids"], targets["ids"][index*2:index*2+2])
        assert torch.allclose(t["teacher"], targets["teacher"][index*2:index*2+2], atol=1e-6)
        part.update(mode="train", direction="reverse", ids=t["ids"], teacher=t["teacher"])
        separate_loss = separate_loss + other(seq[:, :hi], 2, opsd_request=part)["kl"].sum() / 4
    separate_loss.backward()
    assert torch.allclose(loss, separate_loss, atol=1e-6)
    for p, q in zip(actor.parameters(), other.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)
    assert all(p.grad is None for p in teacher.parameters())


def test_missing_or_malformed_targets_fail_closed():
    with pytest.raises(RuntimeError, match="fallback forbidden"):
        selected_token_fallback_forbidden()
    h, w = torch.ones(1, 2), torch.ones(6, 2)
    with pytest.raises(ValueError, match="Duplicate"):
        distribution_kl(h, w, torch.tensor([[1, 1]]), torch.tensor([[.3, .3, .4]]).log(), torch.tensor([1]))
    with pytest.raises(ValueError, match="sum to one"):
        distribution_kl(h, w, torch.tensor([[1, 2]]), torch.tensor([[.3, .3, .3]]).log(), torch.tensor([1]))


def test_topk_installer_preserves_actual_production_paths():
    from opencode_rl.install_training_fixes import transform
    root = source_root()
    source = (root / "skyrl/backends/skyrl_train/workers/worker.py").read_text()
    frozen = transform("worker", source)
    extended = extend_topk_source("worker", frozen)
    assert "prefix_forward_backward" in extended and "mm_token_type_ids=experience.mm_token_type_ids" in extended
    assert extended.index("return forward_backward(self, data") < extended.index("return prefix_forward_backward(self, data")
    tree = ast.parse(extended)
    for cls_name, method in (("PolicyWorkerBase", "forward"), ("RefWorkerBase", "forward")):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
        assert "opsd_topk" in ast.unparse(node.body[1])
    fused = source_file("OPSD_TEST_FUSED_CE_SOURCE").read_text()
    patched = extend_topk_source("fused", transform("fused", fused))
    assert "response_suffix_linear_ce" in patched and "_full_weight(head.weight)" in patched
    assert patched.index("return project_response(") < patched.index("response_suffix_linear_ce(")
    with pytest.raises(ValueError, match="Already"):
        extend_topk_source("fused", patched)


def make_worker(actor):
    algorithm = NS(temperature=1., policy_loss_type=LOSS_NAME, use_kl_loss=False, use_entropy_loss=False)
    return NS(model=actor, cfg=NS(algorithm=algorithm, micro_forward_batch_size_per_gpu=1,
              micro_train_batch_size_per_gpu=1, max_tokens_per_microbatch=0),
              _opsd_test_device="cpu", optimizer=None, mesh_rank=NS(dp_size=1),
              device_mesh=NS(get_group=lambda _: None), scheduler=NS(get_last_lr=lambda: [1e-6]),
              strategy=NS(backward=lambda loss, *_: loss.backward(), all_reduce=lambda values, **_: values))


@pytest.mark.parametrize("direction", ["reverse", "forward"])
def test_actual_worker_transport_backward_and_no_sampled_policy_loss(direction):
    from examples.train.answer_self_distillation import topk_worker
    torch.manual_seed(18)
    actor, teacher = TinyActor(), TinyActor()
    seq = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]])
    data = TrainingInputBatch({"sequences": seq, "attention_mask": torch.ones_like(seq),
                              "mm_token_type_ids": torch.zeros_like(seq), "loss_mask": mask,
                              "response_mask": mask.clone()})
    data.metadata = {"response_length": 3, "opsd_topk": config()}
    data.metadata["opsd_topk"].update(loss=f"teacher_topk_tail_{direction}_kl", kl_direction=direction)
    output = topk_worker.forward(make_worker(teacher), data, "teacher")
    assert len(output.loss_fn_outputs) == 2
    data["opsd_ids"] = TensorList([r["ids"] for r in output.loss_fn_outputs])
    data["opsd_teacher"] = TensorList([r["teacher"] for r in output.loss_fn_outputs])
    score = topk_worker.forward(make_worker(actor), data, "score")
    data["advantages"] = mask.float() / mask.sum()
    data.metadata["opsd_topk_normalized"] = True
    trained = topk_worker.forward_backward(make_worker(actor), data)
    expected = sum(r["kl"].sum() for r in score.loss_fn_outputs) / mask.sum()
    assert trained.metrics["policy_loss"] == pytest.approx(expected.item(), abs=1e-6)
    assert trained.metrics["opsd/action_tokens"] == 4
    assert trained.metrics["opsd/action_importance_weighting"] == 0
    assert trained.metrics["opsd/forward_kl"] == float(direction == "forward")
    assert actor.weight.grad.abs().sum() > 0 and teacher.weight.grad is None
    with pytest.raises(ValueError, match="change objectives"):
        topk_worker.forward_backward(make_worker(actor), data, loss_fn="rollout_is")


def test_sparse_targets_survive_tensorlist_transport_and_dp_padding():
    from examples.train.answer_self_distillation.topk_worker import _request
    data = TrainingInputBatch({"sequences": torch.zeros(2, 5, dtype=torch.long),
        "loss_mask": torch.tensor([[1, 0, 1], [0, 0, 1]]),
        "opsd_ids": TensorList([torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7, 8, 9]])]),
        "opsd_teacher": TensorList([torch.ones(2, 4).log_softmax(-1), torch.ones(1, 4).log_softmax(-1)])})
    data.metadata = {"opsd_topk": config()}
    padded = pad_training_input_batch(data, 2)
    indices = torch.tensor([1, 0, 2, 3])
    selected = TrainingInputBatch({key: value[indices] for key, value in padded.items()})
    selected.metadata = padded.metadata
    req = _request(selected, config(), "train")
    assert req["ids"].tolist() == [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
    assert req["teacher"].shape == (3, 4)


def test_topk_global_normalization_has_no_reward_or_group_dependence():
    trainer = AnswerOpsdTrainerMixin.__new__(AnswerOpsdTrainerMixin)
    trainer.opsd_plan = {"opsd": config()}
    data = TrainingInputBatch({"loss_mask": torch.tensor([[1, 0, 1], [1, 1, 1]])})
    data.metadata = {}
    data["advantages"] = data["loss_mask"].float()
    trainer._normalize_advantages(data, [(0, 2)])
    assert data["advantages"].sum() == 1 and data.metadata["opsd_topk_normalized"]
    with pytest.raises(ValueError, match="once"):
        trainer._normalize_advantages(data, [(0, 2)])


def test_trainer_teacher_only_hint_sparse_transport_and_turn_audit():
    from examples.train.answer_self_distillation.tests.test_opsd import batch, Tokenizer
    from examples.train.answer_self_distillation import topk_worker
    torch.manual_seed(37)
    actor, teacher = TinyActor(), TinyActor()
    trainer = AnswerOpsdTrainerMixin.__new__(AnswerOpsdTrainerMixin)
    trainer.turn_demonstrations = None
    trainer._check_ocrl_configuration = lambda: None
    trainer.opsd_plan = {"opsd": {**config(), "teacher_max_context": 32}}
    trainer.tokenizer, trainer.all_metrics = Tokenizer(), {}
    trainer._hint_ids = {"t1": [60], "t2": [70, 71]}
    trainer.has_critic, trainer.ref_model = False, object()
    captured = {}
    def forward(role, data):
        captured[role] = data
        return topk_worker.forward(make_worker(teacher if role == "ref" else actor), data,
                                   "teacher" if role == "ref" else "score")
    trainer.dispatch = NS(forward=forward, empty_cache=lambda *args: None)
    data = batch()
    data["rollout_logprobs"] = torch.ones(2, 6) * -4
    trainer.fwd_logprobs_values_reward(data)
    assert captured["policy"]["sequences"] is data["sequences"]
    assert 60 not in captured["policy"]["sequences"] and 60 in captured["ref"]["sequences"]
    assert torch.equal(captured["policy"]["mm_token_type_ids"], data["mm_token_type_ids"])
    assert captured["policy"]["pixel_values"] is data["pixel_values"]
    assert data["opsd_ids"][0].shape == (4, 3) and data["opsd_teacher"][1].shape == (4, 4)
    data["rewards"] = torch.ones(2, 6) * 9999
    trainer.apply_reward_kl_penalty(data)
    assert not data["rewards"].any() and data.metadata["answer_opsd_guidance_ready"]
    rows = per_turn_audit(data, NS(get=lambda _: {"stage": "s_3", "answer_sha256": "test"}))
    assert rows[0]["outcome_reward"] == 0 and rows[0]["outcome_loss_weight"] == 0
    assert [turn["action_tokens"] for turn in rows[0]["turns"]] == [2, 2]
    assert sum(t["topk_kl_sum"] for t in rows[0]["turns"]) == pytest.approx(data["opsd_kl"][0].sum().item())


def test_real_fused_source_executes_distribution_head_not_sampled_ce(monkeypatch):
    import os
    import sys
    import types
    from opencode_rl.install_training_fixes import transform
    source = source_file("OPSD_TEST_FUSED_CE_SOURCE").read_text()
    source = extend_topk_source("fused", transform("fused", source))
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_fused_forward")
    stub = types.ModuleType("test_linear_ce")
    stub.LinearCrossEntropy = NS(apply=selected_token_fallback_forbidden)
    monkeypatch.setitem(sys.modules, "skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton", stub)
    namespace = {"torch": torch, "os": os, "_full_weight": lambda w: w}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<actual-transformed-fused-forward>", "exec"), namespace)
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.lm_head = torch.nn.Embedding(16, 4), torch.nn.Linear(4, 16, bias=False)
            self.config = NS(image_token_id=15)
        def forward(self, sequences, **kwargs):
            return {"logits": self.lm_head(self.emb(sequences))}
    backbone = Backbone()
    wrapper = NS(model=backbone, is_vlm=True, sequence_parallel_size=1, remove_microbatch_padding=False)
    seq = torch.tensor([[1, 2, 3, 4, 5]])
    req = {"mode": "teacher", "mask": torch.tensor([[1, 0, 1]], dtype=torch.bool), "k": 3, "chunk_size": 1}
    result = namespace["_fused_forward"](wrapper, seq, 3, attention_mask=torch.ones_like(seq),
        mm_token_type_ids=torch.zeros_like(seq), opsd_request=req)
    assert result["ids"].shape == (2, 3) and result["teacher"].shape == (2, 4)
    req.update(mode="train", direction="reverse", ids=result["ids"], teacher=result["teacher"])
    # Perturb after taking the fixed teacher snapshot so gradients are nonzero.
    with torch.no_grad():
        backbone.lm_head.weight[0] += 1
    scored = namespace["_fused_forward"](wrapper, seq, 3, attention_mask=torch.ones_like(seq),
        mm_token_type_ids=torch.zeros_like(seq), opsd_request=req)
    scored["kl"].sum().backward()
    assert backbone.lm_head.weight.grad.abs().sum() > 0


def test_backward_retains_no_sequence_by_vocabulary_logits():
    h = torch.randn(17, 6, requires_grad=True)
    w = torch.randn(101, 6, requires_grad=True)
    actions = torch.arange(17)
    ids, teacher, _ = teacher_targets(h.detach() + .1, w.detach(), actions, 5, 3)
    shapes = []
    def pack(t):
        shapes.append(tuple(t.shape))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = distribution_kl(h, w, ids, teacher, actions, 3)[0]
    assert (17, 101) not in shapes and (3, 101) not in shapes
    assert set(shapes) == {(17, 6), (101, 6), (17, 5)}  # target has K+1=6
    result.sum().backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(w.grad).all()


def test_simulated_unequal_dp_partition_matches_global_token_mean():
    torch.manual_seed(43)
    h = torch.randn(7, 4, dtype=torch.double)
    w = torch.randn(11, 4, dtype=torch.double)
    ids, teacher, _ = teacher_targets(h + .3, w, torch.arange(7), 3)
    full = w.clone().requires_grad_()
    distribution_kl(h, full, ids, teacher, torch.arange(7))[0].mean().backward()
    grads = []
    for lo, hi in ((0, 2), (2, 7)):
        rank_weight = w.clone().requires_grad_()
        loss = distribution_kl(h[lo:hi], rank_weight, ids[lo:hi], teacher[lo:hi], torch.arange(lo, hi))[0].sum() / 7
        (loss * 2).backward()  # worker correction before FSDP's DP average
        grads.append(rank_weight.grad)
    assert torch.allclose(sum(grads) / 2, full.grad, atol=1e-10)


def test_token_budget_microbatch_reordering_keeps_teacher_row_identity():
    from examples.train.answer_self_distillation import topk_worker
    torch.manual_seed(53)
    actor = TinyActor()
    seq = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [9, 8, 7, 6, 5, 4]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    mask = torch.tensor([[1, 1], [0, 1], [1, 0]])
    data = TrainingInputBatch({"sequences": seq, "attention_mask": attention,
        "loss_mask": mask, "response_mask": mask, "mm_token_type_ids": torch.zeros_like(seq)})
    data.metadata = {"response_length": 2, "opsd_topk": config()}
    sample_worker = make_worker(actor)
    expected = topk_worker.forward(sample_worker, data, "teacher")
    sample_worker.cfg.max_tokens_per_microbatch = 8
    actual = topk_worker.forward(sample_worker, data, "teacher")
    for a, b in zip(actual.loss_fn_outputs, expected.loss_fn_outputs):
        assert torch.equal(a["ids"], b["ids"])
        assert torch.allclose(a["teacher"], b["teacher"], atol=1e-6)
