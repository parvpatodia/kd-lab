"""Tests for the hard-label (B0/B1) supervised distiller. CPU-only, deterministic."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kd_lab.distillation.on_policy import Rollouts, Sampler, StudentRolloutSource
from kd_lab.distillation.supervised import SupervisedDistiller, cross_entropy_next_token


class _ToyLM(torch.nn.Module):
    def __init__(self, vocab: int = 11, dim: int = 8) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))


class _FixedSampler(Sampler):
    def __init__(self, rollouts: Rollouts) -> None:
        self._r = rollouts

    def generate(self, model, prompts) -> Rollouts:
        return self._r


class TestCrossEntropyNextToken:
    def test_perfect_prediction_is_zero(self):
        # logits put almost all mass on the correct next token at each scored position.
        ids = torch.tensor([[0, 1, 2, 3]])
        rmask = torch.tensor([[0, 1, 1, 1]])  # response = tokens 1,2,3
        logits = torch.zeros(1, 4, 5)
        logits[0, 0, 1] = 50.0  # predict id 1
        logits[0, 1, 2] = 50.0  # predict id 2
        logits[0, 2, 3] = 50.0  # predict id 3
        assert cross_entropy_next_token(logits, ids, rmask).item() < 1e-3

    def test_ignores_masked_positions(self):
        # changing a label at a masked-out (shifted) position must not change the loss.
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(1, 4, 11, generator=g)
        rmask = torch.tensor([[0, 0, 1, 1]])  # shifted mask [0,1,1]: shift-pos 0 is masked out
        a = cross_entropy_next_token(logits, torch.tensor([[0, 1, 2, 3]]), rmask)
        b = cross_entropy_next_token(logits, torch.tensor([[0, 9, 2, 3]]), rmask)  # masked-out label
        assert torch.equal(a, b)

    def test_matches_manual_masked_mean(self):
        g = torch.Generator().manual_seed(1)
        logits = torch.randn(2, 5, 7, generator=g)
        ids = torch.randint(0, 7, (2, 5), generator=g)
        rmask = torch.tensor([[0, 1, 1, 0, 1], [0, 0, 1, 1, 1]])
        got = cross_entropy_next_token(logits, ids, rmask)

        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        mask = rmask[:, 1:].float()
        ce = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, 7), shift_labels.reshape(-1), reduction="none"
        ).view_as(shift_labels)
        manual = (ce * mask).sum() / mask.sum()
        assert torch.allclose(got, manual, atol=1e-6)


class TestSupervisedDistiller:
    def _rollout(self, vocab=11):
        ids = torch.randint(0, vocab, (3, 6))
        attn = torch.ones_like(ids)
        rmask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]])
        return Rollouts(input_ids=ids, attention_mask=attn, response_mask=rmask)

    def test_step_updates_student(self):
        torch.manual_seed(0)
        student = _ToyLM()
        roll = self._rollout()
        src = StudentRolloutSource(student, _FixedSampler(roll))
        opt = torch.optim.SGD(student.parameters(), lr=0.5)
        before = [p.detach().clone() for p in student.parameters()]

        dist = SupervisedDistiller(student, src, opt, device="cpu", grad_clip=1.0)
        out = dist.step(prompts=None)

        assert math.isfinite(out["loss"])
        assert out["batch_size"] == 3
        assert out["n_response_tokens"] == int(roll.response_mask[:, 1:].sum())
        assert any(not torch.equal(a, b) for a, b in zip(before, student.parameters(), strict=True))

    def test_loss_decreases(self):
        torch.manual_seed(1)
        student = _ToyLM(vocab=9)
        ids = torch.randint(0, 9, (4, 5))
        attn = torch.ones_like(ids)
        rmask = torch.ones_like(ids)
        rmask[:, 0] = 0
        roll = Rollouts(input_ids=ids, attention_mask=attn, response_mask=rmask)
        src = StudentRolloutSource(student, _FixedSampler(roll))
        opt = torch.optim.SGD(student.parameters(), lr=0.5)
        dist = SupervisedDistiller(student, src, opt, device="cpu", grad_clip=0.0)

        losses = [dist.step(prompts=None)["loss"] for _ in range(30)]
        assert losses[-1] < losses[0] - 1e-3
