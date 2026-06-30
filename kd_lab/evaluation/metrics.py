"""Evaluation metrics for the on-policy distillation study.

Pure functions on results (token-id lists, correctness arrays), so they are deterministic
and unit-testable without a model. ``pass_at_k`` is the unbiased estimator of Chen et al.
(2021) and is verified against brute-force enumeration in tests.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem with ``c`` correct out of ``n`` samples.

    pass@k = 1 - C(n - c, k) / C(n, k), computed in the numerically stable product form.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def aggregate_pass_at_k(num_correct_per_problem: Sequence[int], n: int, k: int) -> float:
    """Mean pass@k over problems, each with ``n`` samples and a per-problem correct count."""
    return float(np.mean([pass_at_k(n, int(c), k) for c in num_correct_per_problem]))


def distinct_n(sequences: Sequence[Sequence], n: int = 1) -> float:
    """Distinct-n: unique n-grams divided by total n-grams across all sequences."""
    total = 0
    uniq: set = set()
    for seq in sequences:
        grams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return (len(uniq) / total) if total else 0.0


def empirical_token_entropy(sequences: Sequence[Sequence], base: float = 2.0) -> float:
    """Shannon entropy (in ``base``) of the empirical unigram distribution of tokens.

    A diversity-collapse signal for the reverse-KL failure-mode test (H2). For predictive
    entropy from model logits, compute it at generation time; this version needs only ids.
    """
    counts = Counter(tok for seq in sequences for tok in seq)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values()), dtype=float) / total
    return float(-(probs * (np.log(probs) / np.log(base))).sum())


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean. Returns ``(mean, ci_low, ci_high)``."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_diff(
    correct_a: Sequence[float], correct_b: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
):
    """Paired bootstrap CI of the mean per-example difference ``a - b``.

    Use for the headline OPD-vs-off-policy gap: the arrays must be aligned by example
    (same prompt, same order). Returns ``(mean_diff, ci_low, ci_high)``.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must be aligned by example")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    diffs = d[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def horizon_stratified_accuracy(
    records: Sequence[dict], horizon_key: str = "k", correct_key: str = "correct", **ci_kwargs
) -> dict:
    """Accuracy with bootstrap CI per horizon bucket.

    ``records`` is a list of dicts each with a horizon value and a 0/1 correctness. Returns
    ``{horizon: {mean, ci_low, ci_high, n}}`` sorted by horizon. This is the headline RQ1
    table/plot: off-policy accuracy should fall faster as horizon grows than on-policy.
    """
    by: dict = {}
    for r in records:
        by.setdefault(r[horizon_key], []).append(float(r[correct_key]))
    out: dict = {}
    for h in sorted(by):
        mean, lo, hi = bootstrap_ci(by[h], **ci_kwargs)
        out[h] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(by[h])}
    return out
