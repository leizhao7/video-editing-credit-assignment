"""Regression against sources fetched from the actual failed Modal image."""
from examples.train.answer_self_distillation.tests.source_fixtures import source_file
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from examples.train.answer_self_distillation import topk_worker
from examples.train.answer_self_distillation.tests.test_topk import config, make_worker
from examples.train.answer_self_distillation.tests.test_turn_demonstrations import fixture, PaddedActor, Tokenizer
from examples.train.answer_self_distillation.turn_teacher import make_turn_batch


def load_source(name, path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Frozen production source changed"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("count", [1, 2, 3, 5, 7, 8, 16])
def test_four_rank_padding_with_actual_production_token_iterator(monkeypatch, direction, count):
    from skyrl.backends.skyrl_train.workers import worker_utils
    root = source_file("OPSD_TEST_FROZEN_SOURCE_DIR", directory=True)
    trim = load_source("failed_image_trim", root / "microbatch_trim.py",
        "6414b1672216387606653d77a1da970374cd2ae05f83ea9c5a4c4d01e5ddfd77")
    native = load_source("failed_image_iterators", root / "worker_utils.py",
        "fdf632839333ef5ecd5c07597156e4e9ccc8076c0a2377809fa49b761d971e2d")
    token = native.TokenBasedBatchIterator
    monkeypatch.setattr(token, "_s12_stock_init", token.__init__, raising=False)
    monkeypatch.setattr(token, "_s12_stock_create_microbatch", token._create_microbatch_from_indices, raising=False)
    monkeypatch.setattr(token, "__init__", trim._token_iterator_init)
    monkeypatch.setattr(token, "_create_microbatch_from_indices", trim._token_microbatch)
    # Maximum equalized count exercises real-bin splitting on every shard.
    monkeypatch.setattr(token, "_sync_num_microbatches", lambda self: self.data.batch_size)
    monkeypatch.setattr(worker_utils, "get_microbatch_iterator", native.get_microbatch_iterator)
    data, source_jobs = fixture()
    jobs = [copy.deepcopy(source_jobs[i % len(source_jobs)]) for i in range(count)]
    batch = make_turn_batch(data, jobs, Tokenizer(), 32, merge_size=1)
    batch.metadata["opsd_topk"] = {**config(), "loss": f"teacher_topk_tail_{direction}_kl", "kl_direction": direction}
    actor, observed, target_count = PaddedActor(), {}, 0
    for rank in range(4):
        indices = list(range(rank, len(batch), 4))
        shard = trim._select_rows(batch, indices)
        worker = make_worker(actor)
        worker.cfg.max_tokens_per_microbatch = 16
        output = topk_worker.forward(worker, shard, "teacher")
        for index, result in zip(indices, output.loss_fn_outputs):
            observed[index] = result
            target_count += len(result["ids"])
            assert torch.isfinite(result["teacher"]).all()
            assert len(result["ids"]) == int(batch["loss_mask"][index].sum())
            if index >= count:
                assert result["ids"].shape == (0, 3)
    assert set(observed) == set(range(len(batch)))
    assert target_count == int(batch["loss_mask"].sum())
    # Removing only left padding cannot change any real teacher target.
    for index in range(count):
        full = trim._select_rows(batch, [index])
        expected = topk_worker._head(make_worker(actor), full,
            topk_worker._request(full, batch.metadata["opsd_topk"], "teacher"))
        assert torch.equal(observed[index]["ids"], expected["ids"])
        assert torch.allclose(observed[index]["teacher"], expected["teacher"], atol=1e-6)


def test_failed_teacher_does_not_queue_cleanup_on_broken_workers():
    from examples.train.answer_self_distillation.trainer import AnswerOpsdTrainerMixin
    data, _ = fixture()
    trainer = AnswerOpsdTrainerMixin.__new__(AnswerOpsdTrainerMixin)
    trainer._check_ocrl_configuration = lambda: None
    trainer.has_critic, trainer.ref_model = False, object()
    trainer.all_metrics, trainer.tokenizer = {}, Tokenizer()
    trainer.opsd_plan = {"opsd": {"loss": "teacher_topk_tail_reverse_kl", "demonstrator": {}}}
    error, cleanup = ValueError("original distributed forward failure"), []
    def fail(_): raise error
    trainer.turn_demonstrations = NS(prepare=fail)
    trainer.dispatch = NS(empty_cache=lambda *args: cleanup.append(args))
    with pytest.raises(ValueError) as caught:
        trainer.fwd_logprobs_values_reward(data)
    assert caught.value is error and cleanup == []
