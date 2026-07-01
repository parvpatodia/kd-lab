"""Token-level divergences between a frozen teacher and a trainable student.

These are the loss functions for both off-policy KD (forward KL on teacher data) and
on-policy distillation (reverse KL / JSD on student rollouts). The design separates two
axes, following Agarwal et al. (2023, arXiv:2306.13649):

  * the divergence (forward KL, reverse KL, or generalized JSD), implemented here;
  * the data source (teacher data vs student rollouts vs a mix), implemented in
    ``kd_lab.distillation.on_policy``.

Numerical properties are verified in ``tests/test_scaffold.py``:
  FKL(p, p) == 0, RKL(p, p) == 0, KL >= 0, JSD(0) == JSD(1) == 0, JSD symmetric at
  beta = 0.5 and bounded by ln 2, and reverse != forward on asymmetric inputs.

All math is in log-space for stability. The teacher distribution is detached so no
gradient leaks into the frozen teacher even if a caller forgets ``torch.no_grad``.
"""

from __future__ import annotations

import abc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Divergence(nn.Module, abc.ABC):
    """Base class. Subclasses implement ``_per_token`` on log-probabilities.

    The base handles temperature, teacher detachment, response masking and reduction
    (template-method pattern), so each subclass is a few lines of pure math.

    Args:
        temperature: softmax temperature tau applied to both logits. Default 1.0
            (on-policy distillation uses the raw distributions). Classic Hinton KD uses
            tau > 1.
        scale_grad_by_temp_sq: multiply the loss by tau**2 when tau != 1, the Hinton
            convention that keeps gradient magnitudes comparable across temperatures.
    """

    def __init__(self, temperature: float = 1.0, scale_grad_by_temp_sq: bool = True) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.scale_grad_by_temp_sq = bool(scale_grad_by_temp_sq)

    @abc.abstractmethod
    def _per_token(self, log_p_teacher: torch.Tensor, log_q_student: torch.Tensor) -> torch.Tensor:
        """Return per-position divergence ``[B, T]`` from log-probs ``[B, T, V]``."""
        raise NotImplementedError

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "mean_over_tokens",
    ) -> torch.Tensor:
        """Compute the masked, reduced divergence.

        Args:
            student_logits: ``[B, T, V]`` (gradients flow here).
            teacher_logits: ``[B, T, V]`` (detached internally).
            mask: ``[B, T]`` boolean/0-1; True marks positions to include (response
                tokens). If None, all positions are used.
            reduction: ``"mean_over_tokens"`` (scalar, mean over masked positions) or
                ``"none"`` (return the ``[B, T]`` per-token tensor).
        """
        tau = self.temperature
        log_p = F.log_softmax(teacher_logits.detach() / tau, dim=-1)
        log_q = F.log_softmax(student_logits / tau, dim=-1)
        per_tok = self._per_token(log_p, log_q)  # [B, T]
        if tau != 1.0 and self.scale_grad_by_temp_sq:
            per_tok = per_tok * (tau * tau)
        if reduction == "none":
            return per_tok
        if reduction != "mean_over_tokens":
            raise ValueError(f"unknown reduction: {reduction}")
        if mask is None:
            return per_tok.mean()
        m = mask.to(per_tok.dtype)
        denom = m.sum().clamp_min(1.0)
        return (per_tok * m).sum() / denom


class ForwardKL(Divergence):
    """KL(teacher || student): mass-covering. The Hinton / off-policy KD divergence."""

    def _per_token(self, log_p_teacher: torch.Tensor, log_q_student: torch.Tensor) -> torch.Tensor:
        p = log_p_teacher.exp()
        return (p * (log_p_teacher - log_q_student)).sum(dim=-1)


class ReverseKL(Divergence):
    """KL(student || teacher): mode-seeking. Primary divergence for on-policy distillation
    (MiniLLM, Gu et al. 2023; Thinking Machines, 2025)."""

    def _per_token(self, log_p_teacher: torch.Tensor, log_q_student: torch.Tensor) -> torch.Tensor:
        q = log_q_student.exp()
        return (q * (log_q_student - log_p_teacher)).sum(dim=-1)


class GeneralizedJSD(Divergence):
    """Generalized Jensen-Shannon divergence with mixing weight ``beta``.

        m = beta * teacher + (1 - beta) * student
        JSD(beta) = beta * KL(teacher || m) + (1 - beta) * KL(student || m)

    Properties (tested): JSD(0) == JSD(1) == 0, symmetric in (teacher, student) at
    beta = 0.5, bounded above by ln 2. Note it is NOT equal to forward/reverse KL at the
    endpoints; treat beta as a stability-vs-sharpness knob and cite Agarwal et al. (2023)
    for the generalized treatment.
    """

    def __init__(self, beta: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta must be in [0, 1]")
        self.beta = float(beta)

    def _per_token(self, log_p_teacher: torch.Tensor, log_q_student: torch.Tensor) -> torch.Tensor:
        beta = self.beta
        if beta <= 0.0 or beta >= 1.0:
            return torch.zeros(
                log_p_teacher.shape[:-1], device=log_p_teacher.device, dtype=log_p_teacher.dtype
            )
        p = log_p_teacher.exp()
        q = log_q_student.exp()
        # log m = logaddexp(log(beta) + log_p, log(1 - beta) + log_q), stable in log-space.
        log_m = torch.logaddexp(log_p_teacher + math.log(beta), log_q_student + math.log(1.0 - beta))
        kl_pm = (p * (log_p_teacher - log_m)).sum(dim=-1)
        kl_qm = (q * (log_q_student - log_m)).sum(dim=-1)
        return beta * kl_pm + (1.0 - beta) * kl_qm


def build_divergence(name: str, temperature: float = 1.0, beta: float = 0.5) -> Divergence:
    """Factory used by configs. ``name`` in {reverse_kl, forward_kl, generalized_jsd}."""
    key = name.lower()
    if key in ("rkl", "reverse_kl"):
        return ReverseKL(temperature=temperature)
    if key in ("fkl", "forward_kl"):
        return ForwardKL(temperature=temperature)
    if key in ("jsd", "generalized_jsd"):
        return GeneralizedJSD(beta=beta, temperature=temperature)
    raise ValueError(f"unknown divergence: {name}")
