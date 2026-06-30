"""Positional teacher-student KL probe: the exposure-bias mechanism diagnostic.

Measures the token-level KL between teacher and student over a set of student rollouts, as a
function of the within-sequence response position (0 = first generated token). The exposure-
bias prediction (H1): an off-policy student diverges from the teacher more as position grows
(errors compound and it drifts into states it never trained on), while an on-policy student
stays flatter because it trained on its own state distribution.

The model forward returning ``.logits`` is the HuggingFace CausalLM convention (marked SEAM).
The KL math and the position bucketing are complete and use the same log-space formulas as
``kd_lab.distillation.divergences``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..distillation.on_policy import Rollouts


@torch.no_grad()
def positional_teacher_student_kl(
    student,
    teacher,
    rollouts: Rollouts,
    device: str = "cuda",
    max_positions: int = 256,
    direction: str = "reverse",
) -> dict:
    """Return mean KL per response-token position over the given rollouts.

    Args:
        student, teacher: models whose forward returns ``.logits`` (SEAM).
        rollouts: student-generated sequences with a ``response_mask``.
        max_positions: cap on the position axis.
        direction: ``"reverse"`` = KL(student || teacher) (matches the OPD loss);
            ``"forward"`` = KL(teacher || student).

    Returns ``{position, mean_kl, count}`` as numpy arrays of length ``max_positions``.
    """
    ids = rollouts.input_ids.to(device)
    attn = rollouts.attention_mask.to(device)
    rmask = rollouts.response_mask.to(device)

    s = student(input_ids=ids, attention_mask=attn).logits  # SEAM
    t = teacher(input_ids=ids, attention_mask=attn).logits  # SEAM

    log_s = torch.log_softmax(s[:, :-1, :], dim=-1)
    log_t = torch.log_softmax(t[:, :-1, :], dim=-1)
    if direction == "reverse":
        per = (log_s.exp() * (log_s - log_t)).sum(dim=-1)  # KL(student || teacher)
    elif direction == "forward":
        per = (log_t.exp() * (log_t - log_s)).sum(dim=-1)  # KL(teacher || student)
    else:
        raise ValueError("direction must be 'reverse' or 'forward'")

    pred_mask = rmask[:, 1:].bool().cpu().numpy()
    per_np = per.float().cpu().numpy()

    sums = np.zeros(max_positions)
    counts = np.zeros(max_positions)
    for b in range(pred_mask.shape[0]):
        pos = 0
        for tpos in range(pred_mask.shape[1]):
            if pred_mask[b, tpos]:
                if pos < max_positions:
                    sums[pos] += per_np[b, tpos]
                    counts[pos] += 1
                pos += 1
    with np.errstate(invalid="ignore"):
        mean_kl = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return {"position": np.arange(max_positions), "mean_kl": mean_kl, "count": counts}
