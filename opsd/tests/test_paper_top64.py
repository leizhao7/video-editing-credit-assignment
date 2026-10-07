"""CPU numerical reference for the paper's exact K=64 setting."""
import pytest
import torch

from examples.train.answer_self_distillation.topk import distribution_kl, teacher_targets


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_forward_top64_tail_token_mean_matches_independent_dense_reference(chunk_size):
    torch.manual_seed(107)
    hidden = (.3 * torch.randn(5, 6, dtype=torch.double)).requires_grad_()
    weight = (.3 * torch.randn(97, 6, dtype=torch.double)).requires_grad_()
    teacher_weight = (.3 * torch.randn(97, 6, dtype=torch.double)).requires_grad_()
    teacher_hidden = hidden.detach() + .15
    actions = torch.tensor([0, 12, 44, 68, 96])
    ids, target, _ = teacher_targets(teacher_hidden, teacher_weight, actions, 64, chunk_size)
    assert ids.shape == (5, 64) and target.shape == (5, 65)
    assert not target.requires_grad

    expected_hidden = hidden.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    student_full = (expected_hidden @ expected_weight.T).softmax(-1)
    teacher_full = (teacher_hidden @ teacher_weight.detach().T).softmax(-1)
    complement = torch.ones_like(student_full, dtype=torch.bool).scatter_(1, ids, False)
    student_bins = torch.cat((student_full.gather(1, ids),
                              student_full.masked_fill(~complement, 0).sum(-1, keepdim=True)), -1)
    teacher_bins = torch.cat((teacher_full.gather(1, ids),
                              teacher_full.masked_fill(~complement, 0).sum(-1, keepdim=True)), -1)
    expected = (teacher_bins * (teacher_bins.log() - student_bins.log())).sum(-1).mean()
    actual = distribution_kl(hidden, weight, ids, target, actions, chunk_size, "forward")[0].mean()
    expected.backward()
    actual.backward()
    assert torch.allclose(student_bins.sum(-1), torch.ones(5, dtype=torch.double), atol=1e-12)
    assert (student_bins[:, -1] > 0).all()
    assert (student_bins[:, :-1].sum(-1) < 1).all()
    assert torch.allclose(actual, expected, atol=1e-11)
    assert torch.allclose(hidden.grad, expected_hidden.grad, atol=1e-10)
    assert torch.allclose(weight.grad, expected_weight.grad, atol=1e-10)
    assert teacher_weight.grad is None
