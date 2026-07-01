"""Test suite for the on-policy distillation scaffold.

Divergence tests need torch and are skipped if it is absent. The pointer-chase and metrics
tests are pure-Python and always run. The numerical properties checked here were verified
against a numpy reference and brute-force enumeration during development.
"""

from __future__ import annotations

import math
from itertools import combinations

import pytest

from kd_lab.evaluation.metrics import (
    aggregate_pass_at_k,
    bootstrap_ci,
    distinct_n,
    empirical_token_entropy,
    horizon_stratified_accuracy,
    paired_bootstrap_diff,
    pass_at_k,
)
from kd_lab.tasks.pointer_chase import (
    PointerChaseConfig,
    make_dataset,
    make_eval_sets,
    make_example,
    parse_chain,
    parse_final_answer,
    score_final,
    score_per_hop,
)

try:
    import torch

    from kd_lab.distillation.divergences import (
        ForwardKL,
        GeneralizedJSD,
        ReverseKL,
        build_divergence,
    )
    from kd_lab.distillation.on_policy import shift_for_next_token

    HAS_TORCH = True
except Exception:  # torch not installed: skip only the divergence tests
    HAS_TORCH = False


# --------------------------------------------------------------------------------------
# Divergences
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_TORCH, reason="divergence tests require torch")
class TestDivergences:
    def _logits(self, seed=0, shape=(4, 5, 17)):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)

    def test_self_divergence_is_zero(self):
        t, _ = self._logits()
        assert ForwardKL()(t, t).abs().item() < 1e-5
        assert ReverseKL()(t, t).abs().item() < 1e-5

    def test_kl_non_negative(self):
        t, s = self._logits()
        assert ForwardKL()(s, t).item() > -1e-6
        assert ReverseKL()(s, t).item() > -1e-6

    def test_jsd_endpoints_zero(self):
        t, s = self._logits()
        assert GeneralizedJSD(beta=0.0)(s, t).abs().item() < 1e-6
        assert GeneralizedJSD(beta=1.0)(s, t).abs().item() < 1e-6

    def test_jsd_symmetric_and_bounded(self):
        t, s = self._logits()
        ab = GeneralizedJSD(beta=0.5)(s, t, reduction="none")
        ba = GeneralizedJSD(beta=0.5)(t, s, reduction="none")
        assert torch.allclose(ab, ba, atol=1e-6)
        assert ab.max().item() <= math.log(2) + 1e-5

    def test_reverse_differs_from_forward(self):
        t, s = self._logits()
        f = ForwardKL()(s, t, reduction="none")
        r = ReverseKL()(s, t, reduction="none")
        assert (f - r).abs().max().item() > 1e-3

    def test_masking_only_response_tokens(self):
        t, s = self._logits(shape=(2, 4, 5))
        mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]])
        per = ReverseKL()(s, t, reduction="none")
        manual = (per * mask).sum() / mask.sum()
        assert torch.allclose(ReverseKL()(s, t, mask=mask), manual, atol=1e-6)

    def test_gradient_flows_to_student_not_teacher(self):
        t, s = self._logits()
        s = s.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        ReverseKL()(s, t).backward()
        assert s.grad is not None and s.grad.abs().sum() > 0
        assert t.grad is None  # teacher detached inside the divergence

    def test_temperature_softens(self):
        t, s = self._logits()
        hot = ForwardKL(temperature=4.0, scale_grad_by_temp_sq=False)(s, t)
        cold = ForwardKL(temperature=1.0)(s, t)
        assert hot.item() < cold.item()

    def test_build_divergence(self):
        assert isinstance(build_divergence("reverse_kl"), ReverseKL)
        assert isinstance(build_divergence("forward_kl"), ForwardKL)
        assert isinstance(build_divergence("generalized_jsd", beta=0.3), GeneralizedJSD)
        with pytest.raises(ValueError):
            build_divergence("nope")

    def test_shift_alignment_shapes(self):
        s = torch.randn(2, 6, 11)
        t = torch.randn(2, 6, 11)
        rmask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0]])
        sp, tp, pm = shift_for_next_token(s, t, rmask)
        assert sp.shape == (2, 5, 11) and tp.shape == (2, 5, 11) and pm.shape == (2, 5)
        assert pm.dtype == torch.bool


# --------------------------------------------------------------------------------------
# Pointer-chase task
# --------------------------------------------------------------------------------------
class TestPointerChase:
    def test_deterministic(self):
        assert make_example(7, 4) == make_example(7, 4)

    def test_chain_follows_table_and_length(self):
        cfg = PointerChaseConfig(num_nodes=20)
        ex = make_example(123, 6, cfg)
        import re

        table = {int(a): int(b) for a, b in re.findall(r"(\d+) -> (\d+)", ex["prompt"])}
        chain = ex["chain"]
        assert all(table[chain[i]] == chain[i + 1] for i in range(len(chain) - 1))
        assert ex["answer"] == chain[-1]
        assert len(chain) == ex["k"] + 1

    def test_verifier_round_trips_gold(self):
        ex = make_example(123, 6)
        assert parse_final_answer(ex["target"]) == ex["answer"]
        assert parse_chain(ex["target"]) == ex["chain"]
        assert score_final(ex["target"], ex) is True
        assert score_per_hop(ex["target"], ex) == 1.0

    def test_horizon_dial(self):
        for k in (1, 2, 4, 8):
            assert len(make_example(5, k)["chain"]) == k + 1

    def test_early_error_propagates(self):
        gold = make_example(999, 5)["chain"]
        corrupted = gold[:]
        corrupted[2] = (corrupted[2] + 1) % 32
        assert corrupted[2:] != gold[2:]

    def test_dataset_builders(self):
        train = make_dataset(n=50, ks=[1, 2, 3], base_seed=0)
        assert len(train) == 50 and all(1 <= e["k"] <= 3 for e in train)
        ev = make_eval_sets(n_per_k=10, ks=[1, 2, 4])
        assert set(ev) == {1, 2, 4} and all(len(v) == 10 for v in ev.values())

    def test_score_per_hop_partial(self):
        ex = make_example(1, 4)
        # corrupt the final hop only -> last hop wrong, earlier hops right
        bad_chain = ex["chain"][:]
        bad_chain[-1] = (bad_chain[-1] + 1) % 32
        text = " -> ".join(map(str, bad_chain)) + f"\nFinal: {bad_chain[-1]}"
        assert 0.0 < score_per_hop(text, ex) < 1.0
        assert score_final(text, ex) is False


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
class TestMetrics:
    def _brute_pass_at_k(self, n, c, k):
        items = [1] * c + [0] * (n - c)
        tot = hit = 0
        for comb in combinations(range(n), k):
            tot += 1
            hit += any(items[i] for i in comb)
        return hit / tot

    def test_pass_at_k_matches_brute_force(self):
        for n, c, k in [(8, 2, 3), (10, 4, 2), (6, 1, 4), (16, 3, 4)]:
            assert abs(pass_at_k(n, c, k) - self._brute_pass_at_k(n, c, k)) < 1e-9

    def test_pass_at_k_edges_and_monotonic(self):
        assert pass_at_k(16, 0, 1) == 0.0
        assert pass_at_k(16, 16, 1) == 1.0
        assert pass_at_k(16, 3, 1) <= pass_at_k(16, 3, 4) <= pass_at_k(16, 3, 16)

    def test_aggregate_pass_at_k(self):
        val = aggregate_pass_at_k([0, 8, 16], n=16, k=4)
        assert 0.0 <= val <= 1.0

    def test_distinct_n(self):
        assert distinct_n([[1, 1, 1, 1]], n=1) == pytest.approx(0.25)
        assert distinct_n([[1, 2, 3, 4]], n=1) == pytest.approx(1.0)

    def test_entropy_extremes(self):
        assert empirical_token_entropy([[5, 5, 5, 5]]) == pytest.approx(0.0)
        # two equally frequent tokens -> 1 bit
        assert empirical_token_entropy([[0, 1, 0, 1]]) == pytest.approx(1.0)

    def test_bootstrap_ci_brackets_mean(self):
        mean, lo, hi = bootstrap_ci([0, 1] * 50, seed=0)
        assert lo <= mean <= hi and abs(mean - 0.5) < 0.1

    def test_paired_bootstrap_diff_sign(self):
        a = [1] * 80 + [0] * 20  # 0.80
        b = [1] * 60 + [0] * 40  # 0.60
        mean, lo, hi = paired_bootstrap_diff(a, b, seed=0)
        assert mean == pytest.approx(0.2, abs=1e-9)
        assert lo > 0  # the gap is positive with the CI excluding 0

    def test_paired_requires_alignment(self):
        with pytest.raises(ValueError):
            paired_bootstrap_diff([1, 0, 1], [1, 0])

    def test_horizon_stratified(self):
        recs = [{"k": 1, "correct": 1}, {"k": 1, "correct": 0}, {"k": 2, "correct": 0}]
        out = horizon_stratified_accuracy(recs)
        assert out[1]["n"] == 2 and out[2]["n"] == 1
        assert out[1]["mean"] == pytest.approx(0.5)
