"""Phase 1 hardening tests for the divergence family and the on-policy step.

Extends ``tests/test_scaffold.py`` to the full BUILDPLAN section 6 checklist:

  * reverse-KL gradient *sign* on a 2-token vocab, the one section-6 check the
    delivered suite lacked, cross-checked against the closed form and autograd;
  * a numpy closed-form reference cross-check for forward KL, reverse KL and JSD;
  * the Hinton tau**2 gradient-scaling convention isolated from the softening effect;
  * prompt/pad positions contributing exactly zero to both the loss and the gradient;
  * a CPU end-to-end ``OnPolicyDistiller.step`` on a 2-layer toy model (no GPU, no HF)
    that exercises the ``.logits`` seam, the next-token shift, masking, backward, and
    the optimizer update.

All tests are CPU-only and deterministic. The module is skipped without torch.

The closed form for the reverse-KL gradient (used in the sign test) is

    d/dz_i  KL(q || p)  =  q_i * ( log(q_i / p_i) - KL(q || p) ),   q = softmax(z),

so a token the student over-weights relative to the teacher receives a positive
gradient, and descent pushes its logit down toward the teacher. The gradient also
sums to zero across the vocab because KL(q(z) || p) is invariant to a constant shift
of all logits.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip the whole module if torch is absent

from kd_lab.distillation.divergences import ForwardKL, GeneralizedJSD, ReverseKL
from kd_lab.distillation.on_policy import (
    OnPolicyDistiller,
    Rollouts,
    Sampler,
    StudentRolloutSource,
)


# --------------------------------------------------------------------------------------
# numpy reference implementations (independent of the torch code under test)
# --------------------------------------------------------------------------------------
def _np_log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _np_forward_kl(t_logits: np.ndarray, s_logits: np.ndarray) -> np.ndarray:
    lp, lq = _np_log_softmax(t_logits), _np_log_softmax(s_logits)
    return (np.exp(lp) * (lp - lq)).sum(-1)


def _np_reverse_kl(t_logits: np.ndarray, s_logits: np.ndarray) -> np.ndarray:
    lp, lq = _np_log_softmax(t_logits), _np_log_softmax(s_logits)
    return (np.exp(lq) * (lq - lp)).sum(-1)


def _np_jsd(t_logits: np.ndarray, s_logits: np.ndarray, beta: float) -> np.ndarray:
    lp, lq = _np_log_softmax(t_logits), _np_log_softmax(s_logits)
    p, q = np.exp(lp), np.exp(lq)
    log_m = np.log(beta * p + (1.0 - beta) * q)
    kl_pm = (p * (lp - log_m)).sum(-1)
    kl_qm = (q * (lq - log_m)).sum(-1)
    return beta * kl_pm + (1.0 - beta) * kl_qm


class TestDivergenceNumerics:
    def _logits(self, seed: int, shape=(3, 4, 7)):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)

    def test_matches_numpy_reference(self):
        # forward(student_logits, teacher_logits): the module's first arg is the student.
        t, s = self._logits(3)
        tn, sn = t.numpy(), s.numpy()
        fkl = ForwardKL()(s, t, reduction="none").numpy()
        rkl = ReverseKL()(s, t, reduction="none").numpy()
        jsd05 = GeneralizedJSD(beta=0.5)(s, t, reduction="none").numpy()
        jsd03 = GeneralizedJSD(beta=0.3)(s, t, reduction="none").numpy()
        assert np.allclose(fkl, _np_forward_kl(tn, sn), atol=1e-5)
        assert np.allclose(rkl, _np_reverse_kl(tn, sn), atol=1e-5)
        assert np.allclose(jsd05, _np_jsd(tn, sn, 0.5), atol=1e-5)
        assert np.allclose(jsd03, _np_jsd(tn, sn, 0.3), atol=1e-5)

    def test_reverse_kl_gradient_sign_two_token(self):
        # teacher uniform p=[0.5, 0.5]; student over-weights token 0 (q ~ [0.88, 0.12]).
        teacher = torch.tensor([[[0.0, 0.0]]])
        student = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
        loss = ReverseKL()(student, teacher)  # mask=None -> mean over the single position
        loss.backward()
        grad = student.grad[0, 0]

        q = torch.softmax(student.detach()[0, 0], dim=-1)
        p = torch.softmax(teacher[0, 0], dim=-1)
        rkl = (q * (q.log() - p.log())).sum()
        expected = q * ((q / p).log() - rkl)

        assert torch.allclose(grad, expected, atol=1e-6)  # autograd matches the closed form
        assert grad[0] > 0 and grad[1] < 0  # over-weighted token gets a positive gradient
        assert torch.allclose(grad.sum(), torch.zeros(()), atol=1e-6)  # shift-invariance

    def test_temperature_squared_scaling(self):
        # tau**2 scaling is exactly the ratio between scale_on and scale_off at the same tau,
        # which isolates the Hinton factor from the distribution softening that tau also causes.
        t, s = self._logits(1, shape=(2, 3, 5))
        T = 2.0
        scaled = ForwardKL(temperature=T, scale_grad_by_temp_sq=True)(s, t)
        unscaled = ForwardKL(temperature=T, scale_grad_by_temp_sq=False)(s, t)
        assert torch.allclose(scaled, unscaled * (T * T), atol=1e-6)

    def test_masked_positions_contribute_zero_loss_and_grad(self):
        t, s = self._logits(2, shape=(2, 4, 6))
        s = s.clone().requires_grad_(True)
        mask = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 0]])
        loss = ReverseKL()(s, t, mask=mask)
        loss.backward()

        # masked-out positions contribute exactly zero gradient (prompt/pad invariance).
        assert torch.count_nonzero(s.grad[mask == 0]) == 0
        # and the masked loss equals the manual masked mean over the kept positions.
        per = ReverseKL()(s.detach(), t, reduction="none")
        manual = (per * mask).sum() / mask.sum()
        assert torch.allclose(loss.detach(), manual, atol=1e-6)


# --------------------------------------------------------------------------------------
# CPU end-to-end on-policy step (toy model, no GPU, no HuggingFace)
# --------------------------------------------------------------------------------------
class _ToyLM(torch.nn.Module):
    """Minimal 2-layer LM whose forward returns ``.logits`` (the SEAM convention)."""

    def __init__(self, vocab: int = 11, dim: int = 8) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))


class _FixedSampler(Sampler):
    """Returns a precomputed rollout, so the on-policy step is deterministic and GPU-free."""

    def __init__(self, rollouts: Rollouts) -> None:
        self._r = rollouts

    def generate(self, model, prompts) -> Rollouts:
        return self._r


class TestOnPolicyStepCPU:
    def test_step_updates_student_not_teacher(self):
        torch.manual_seed(0)
        vocab = 11
        student, teacher = _ToyLM(vocab), _ToyLM(vocab)
        ids = torch.randint(0, vocab, (3, 6))
        attn = torch.ones_like(ids)
        rmask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]])
        roll = Rollouts(input_ids=ids, attention_mask=attn, response_mask=rmask)

        src = StudentRolloutSource(student, _FixedSampler(roll))
        opt = torch.optim.SGD(student.parameters(), lr=0.5)
        student_before = [p.detach().clone() for p in student.parameters()]
        teacher_before = [p.detach().clone() for p in teacher.parameters()]

        dist = OnPolicyDistiller(student, teacher, src, ReverseKL(), opt, device="cpu", grad_clip=1.0)
        out = dist.step(prompts=None)

        assert math.isfinite(out["loss"])
        assert out["batch_size"] == 3
        assert out["n_response_tokens"] == int(rmask[:, 1:].sum())  # scored tokens = shifted mask

        assert any(not torch.equal(a, b) for a, b in zip(student_before, student.parameters(), strict=True))
        assert all(torch.equal(a, b) for a, b in zip(teacher_before, teacher.parameters(), strict=True))
        assert all(p.grad is None for p in teacher.parameters())  # teacher frozen, no grad

    def test_loss_decreases_when_student_chases_teacher(self):
        # Repeated steps on a fixed rollout drive the student toward the teacher, so the
        # reverse-KL loss falls over the run (Phase 3 gate: "loss decreases"). Strict
        # per-step monotonicity is not asserted: it is not guaranteed for a finite lr on a
        # non-convex objective.
        torch.manual_seed(1)
        vocab = 9
        student, teacher = _ToyLM(vocab), _ToyLM(vocab)
        ids = torch.randint(0, vocab, (4, 5))
        attn = torch.ones_like(ids)
        rmask = torch.ones_like(ids)
        rmask[:, 0] = 0  # first token is the prompt
        roll = Rollouts(input_ids=ids, attention_mask=attn, response_mask=rmask)

        src = StudentRolloutSource(student, _FixedSampler(roll))
        opt = torch.optim.SGD(student.parameters(), lr=0.5)
        dist = OnPolicyDistiller(student, teacher, src, ReverseKL(), opt, device="cpu", grad_clip=0.0)

        losses = [dist.step(prompts=None)["loss"] for _ in range(30)]
        assert losses[-1] < losses[0] - 1e-3  # the student moves toward the teacher
        assert min(losses[20:]) < min(losses[:5])  # later window is clearly lower
