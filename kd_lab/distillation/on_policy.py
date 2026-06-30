"""On-policy distillation: generate -> score with teacher -> token-level divergence -> update.

Why this is stable and cheap: it is supervised learning on self-generated data. The student
samples a rollout, the rollout is treated as fixed data (no gradient through sampling), and a
frozen teacher supplies a dense per-token target distribution. There is no policy-gradient
term and no REINFORCE variance. This is the language-model analog of DAgger (Ross et al.,
2011) and the GKD recipe (Agarwal et al., 2023).

Repo-integration seams are marked ``# SEAM``: the model forward returns ``.logits`` (the
HuggingFace CausalLM convention; adjust if kd-lab wraps models differently) and the concrete
``Sampler`` (HF ``generate`` or vLLM) is injected. The control flow, the teacher/student
alignment, the next-token shift, and the response masking are complete and verified.
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass

import torch

from .divergences import Divergence


@dataclass
class Rollouts:
    """A batch of prompt+response sequences.

    Attributes:
        input_ids: ``[B, T]`` token ids for prompt followed by response.
        attention_mask: ``[B, T]`` 1 for real tokens, 0 for padding.
        response_mask: ``[B, T]`` 1 for generated (response) tokens, 0 for prompt and pad.
            The divergence is computed only where ``response_mask`` is 1.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor


class Sampler(abc.ABC):
    """Turns prompts into token sequences. Implement with HF ``generate`` or vLLM."""

    @abc.abstractmethod
    def generate(self, model, prompts) -> Rollouts:  # SEAM: signature matches your stack
        raise NotImplementedError


class RolloutSource(abc.ABC):
    """Where the sequences scored by the teacher come from."""

    @abc.abstractmethod
    def rollouts(self, prompts) -> Rollouts:
        raise NotImplementedError


class StudentRolloutSource(RolloutSource):
    """On-policy: sample from the *current* student. No gradient through sampling."""

    def __init__(self, student, sampler: Sampler) -> None:
        self.student = student
        self.sampler = sampler

    @torch.no_grad()
    def rollouts(self, prompts) -> Rollouts:
        was_training = self.student.training
        self.student.eval()
        roll = self.sampler.generate(self.student, prompts)
        if was_training:
            self.student.train()
        return roll


class TeacherRolloutSource(RolloutSource):
    """Off-policy baseline: precomputed teacher generations (or gold sequences)."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset  # SEAM: provides .batch(prompts) -> Rollouts

    def rollouts(self, prompts) -> Rollouts:
        return self.dataset.batch(prompts)


class MixedRolloutSource(RolloutSource):
    """Mix on-policy and off-policy sources by fraction lambda (1.0 = fully on-policy).

    This implements the data-source axis of GKD: lambda = 0 reproduces off-policy KD,
    lambda = 1 is fully on-policy, intermediate values interpolate (RQ3 / H3).
    """

    def __init__(self, on_source: RolloutSource, off_source: RolloutSource, lam: float) -> None:
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lambda must be in [0, 1]")
        self.on_source = on_source
        self.off_source = off_source
        self.lam = float(lam)

    def rollouts(self, prompts) -> Rollouts:
        if random.random() < self.lam:
            return self.on_source.rollouts(prompts)
        return self.off_source.rollouts(prompts)


def shift_for_next_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align logits to next-token prediction targets.

    ``logits[:, t]`` predicts ``token[:, t + 1]``. We score positions that predict a
    response token, so we take logits ``[:, :-1]`` and the response mask shifted to
    ``[:, 1:]``. Verified in tests: only response tokens contribute to the loss.

    Returns ``(student_pred, teacher_pred, pred_mask)`` on the shared ``[B, T-1]`` grid.
    """
    student_pred = student_logits[:, :-1, :]
    teacher_pred = teacher_logits[:, :-1, :]
    pred_mask = response_mask[:, 1:].bool()
    return student_pred, teacher_pred, pred_mask


class OnPolicyDistiller:
    """One optimization step of on-policy distillation.

    Args:
        student: trainable model (HuggingFace CausalLM-style; forward returns ``.logits``).
        teacher: frozen model; set to eval and ``requires_grad_(False)`` in ``__init__``.
        rollout_source: ``StudentRolloutSource`` (on-policy), ``TeacherRolloutSource``
            (off-policy baseline), or ``MixedRolloutSource`` (lambda interpolation).
        divergence: a ``Divergence`` (reverse KL by default for on-policy).
        optimizer: torch optimizer over the student parameters.
        device: training device.
        grad_clip: max grad norm; 0/None disables.
    """

    def __init__(
        self,
        student,
        teacher,
        rollout_source: RolloutSource,
        divergence: Divergence,
        optimizer: torch.optim.Optimizer,
        *,
        device: str = "cuda",
        grad_clip: float = 1.0,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.rollout_source = rollout_source
        self.divergence = divergence
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def step(self, prompts) -> dict:
        """Run generate -> teacher-score -> divergence -> update. Returns metrics."""
        roll = self.rollout_source.rollouts(prompts)  # sampling happens with no grad inside
        ids = roll.input_ids.to(self.device)
        attn = roll.attention_mask.to(self.device)
        rmask = roll.response_mask.to(self.device)

        with torch.no_grad():  # teacher is frozen
            teacher_logits = self.teacher(input_ids=ids, attention_mask=attn).logits  # SEAM
        student_logits = self.student(input_ids=ids, attention_mask=attn).logits  # SEAM; grad flows

        s_pred, t_pred, pred_mask = shift_for_next_token(student_logits, teacher_logits, rmask)
        loss = self.divergence(s_pred, t_pred, mask=pred_mask)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.detach()),
            "n_response_tokens": int(pred_mask.sum()),
            "batch_size": int(ids.shape[0]),
        }
