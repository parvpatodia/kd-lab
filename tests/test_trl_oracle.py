"""Phase 3 correctness oracle: the hand-rolled divergences vs TRL's GKD loss.

TRL's ``GKDTrainer.generalized_jsd_loss`` (arXiv:2306.13649) is the reference. Its convention,
read from the source: beta=0 returns KL(teacher || student) (our ForwardKL), beta=1 returns
KL(student || teacher) (our ReverseKL), and for 0 < beta < 1 it uses the mixture
m = beta*teacher + (1-beta)*student with beta*KL(teacher||m) + (1-beta)*KL(student||m), identical
to our GeneralizedJSD. TRL returns a per-element (over vocab) tensor with reduction="none", so we
sum over the vocab axis to compare with our per-token divergence.

The one intentional difference (DESIGN.md section 2): TRL *special-cases* the endpoints to the raw
KLs, whereas our GeneralizedJSD returns the true generalized JSD, which is 0 at beta in {0, 1}.
The last test pins that difference so it is not "fixed" into a bug.

This is not installed by the ``dev`` extra, so it skips cleanly in CI (trl is in ``experiments``).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
try:
    from trl.experimental.gkd.gkd_trainer import GKDTrainer
except Exception:  # pragma: no cover - version-dependent import path
    pytest.skip("trl GKD trainer unavailable in this version", allow_module_level=True)

from kd_lab.distillation.divergences import ForwardKL, GeneralizedJSD, ReverseKL


def _logits(seed: int, shape=(2, 3, 16)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)  # teacher, student


def _trl(student, teacher, beta):
    # per-token: sum TRL's per-vocab reduction="none" over the vocab axis.
    return GKDTrainer.generalized_jsd_loss(student, teacher, beta=beta, reduction="none").sum(-1)


def test_forward_kl_matches_trl_beta0():
    teacher, student = _logits(0)
    mine = ForwardKL()(student, teacher, reduction="none")
    assert torch.allclose(mine, _trl(student, teacher, 0.0), atol=1e-5)


def test_reverse_kl_matches_trl_beta1():
    teacher, student = _logits(1)
    mine = ReverseKL()(student, teacher, reduction="none")
    assert torch.allclose(mine, _trl(student, teacher, 1.0), atol=1e-5)


def test_generalized_jsd_matches_trl_interior_beta():
    teacher, student = _logits(2)
    for beta in (0.1, 0.3, 0.5, 0.9):
        mine = GeneralizedJSD(beta=beta)(student, teacher, reduction="none")
        assert torch.allclose(mine, _trl(student, teacher, beta), atol=1e-5), beta


def test_jsd_endpoints_differ_from_trl_by_design():
    # Ours -> 0 at the endpoints (true JSD); TRL special-cases them to the raw KLs.
    teacher, student = _logits(3)
    ours0 = GeneralizedJSD(beta=0.0)(student, teacher, reduction="none")
    assert ours0.abs().max().item() < 1e-6
    assert _trl(student, teacher, 0.0).abs().max().item() > 1e-2
