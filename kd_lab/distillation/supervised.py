"""Supervised (hard-label) distillation for the off-policy baselines B0 and B1.

B0 (SFT on gold) and B1 (sequence-level KD; Kim & Rush, 2016, arXiv:1606.07947) both minimize
the cross-entropy of the student on fixed target tokens over the response positions. They differ
only in the data source: B0 trains on gold targets, B1 on the teacher's own generations. Neither
needs a teacher forward pass at train time, so they do not fit the soft-target ``Divergence``
interface (which requires teacher logits). This thin distiller reuses the rollout-source,
next-token shift, and response-mask conventions of ``OnPolicyDistiller``.

B2 (off-policy logit KD; Hinton, 2015) is *not* here: it uses the teacher's soft distribution and
is expressed as ``OnPolicyDistiller`` + ``ForwardKL`` + ``TeacherRolloutSource`` (config-only).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .on_policy import RolloutSource


def cross_entropy_next_token(
    logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Masked next-token cross-entropy over response positions.

    ``logits[:, t]`` predicts ``input_ids[:, t + 1]``, matching ``shift_for_next_token``: score
    positions ``[:, :-1]`` against labels ``[:, 1:]`` and keep only where the (shifted) response
    mask is 1. Averages over the kept positions; prompt and pad contribute exactly zero.

    Args:
        logits: ``[B, T, V]`` student logits (gradients flow here).
        input_ids: ``[B, T]`` token ids; the response tokens are the training targets.
        response_mask: ``[B, T]`` 1 on response tokens, 0 on prompt and pad.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    mask = response_mask[:, 1:].to(logits.dtype)
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view_as(shift_labels)
    denom = mask.sum().clamp_min(1.0)
    return (ce * mask).sum() / denom


class SupervisedDistiller:
    """One optimization step of hard-label distillation (B0 SFT, B1 sequence-level KD).

    Args:
        student: trainable model whose forward returns ``.logits`` (SEAM convention).
        rollout_source: yields ``Rollouts`` whose response tokens are the training targets
            (gold for B0, teacher generations for B1).
        optimizer: torch optimizer over the student parameters.
        device: training device.
        grad_clip: max grad norm; 0/None disables.
    """

    def __init__(
        self,
        student,
        rollout_source: RolloutSource,
        optimizer: torch.optim.Optimizer,
        *,
        device: str = "cuda",
        grad_clip: float = 1.0,
    ) -> None:
        self.student = student
        self.rollout_source = rollout_source
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip

    def step(self, prompts) -> dict:
        """Run rollout -> next-token CE -> update. Returns metrics."""
        roll = self.rollout_source.rollouts(prompts)
        ids = roll.input_ids.to(self.device)
        attn = roll.attention_mask.to(self.device)
        rmask = roll.response_mask.to(self.device)

        logits = self.student(input_ids=ids, attention_mask=attn).logits  # SEAM; grad flows
        loss = cross_entropy_next_token(logits, ids, rmask)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.detach()),
            "n_response_tokens": int(rmask[:, 1:].sum()),
            "batch_size": int(ids.shape[0]),
        }
