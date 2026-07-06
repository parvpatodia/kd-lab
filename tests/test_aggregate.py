"""Tests for results aggregation + plotting. Pure post-processing over the metrics.json schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kd_lab.experiments.aggregate import (
    aggregate_over_seeds,
    condition_key,
    load_runs,
    plot_horizon_comparison,
    plot_positional_kl,
    write_results_table,
)


def _metrics(run_name, method, div, lam, seed, acc_by_k):
    return {
        "plan": {"run_name": run_name, "method": method, "divergence_name": div, "lam": lam, "seed": seed},
        "config_hash": "abc123",
        "horizon_accuracy": {
            str(k): {"mean": a, "ci_low": a - 0.1, "ci_high": a + 0.1, "n": 10} for k, a in acc_by_k.items()
        },
        "positional_kl": {"position": [0, 1, 2], "mean_kl": [0.1, 0.2, 0.3]},
    }


def _write(root: Path, m: dict):
    d = root / m["plan"]["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(m))


def _fixture(root: Path):
    _write(root, _metrics("opd_rkl_seed0", "on_policy", "reverse_kl", 1.0, 0, {2: 0.8, 4: 0.6}))
    _write(root, _metrics("opd_rkl_seed1", "on_policy", "reverse_kl", 1.0, 1, {2: 0.9, 4: 0.5}))
    _write(root, _metrics("b0_sft_seed0", "sft", "forward_kl", 0.0, 0, {2: 0.5, 4: 0.2}))


def test_load_runs_normalizes_horizon_keys(tmp_path):
    _fixture(tmp_path)
    runs = load_runs(str(tmp_path))
    assert len(runs) == 3
    assert all(all(isinstance(k, int) for k in r["horizon_accuracy"]) for r in runs)


def test_aggregate_over_seeds_means(tmp_path):
    _fixture(tmp_path)
    agg = aggregate_over_seeds(load_runs(str(tmp_path)))
    opd = agg["opd_rkl"]  # grouped by run tag (opd_rkl_seed0/1)
    assert opd[2]["n_seeds"] == 2
    assert opd[2]["mean"] == pytest.approx(0.85)  # mean of 0.8 and 0.9
    assert opd[4]["mean"] == pytest.approx(0.55)
    sft = agg["b0_sft"]
    assert sft[2]["n_seeds"] == 1 and sft[2]["std"] == 0.0


def test_write_results_table(tmp_path):
    _fixture(tmp_path)
    out = tmp_path / "table.csv"
    write_results_table(load_runs(str(tmp_path)), str(out))
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 4  # header + 3 runs
    assert "acc_k2" in lines[0] and "acc_k4" in lines[0]


def test_plots_write_files(tmp_path):
    pytest.importorskip("matplotlib")  # figure functions no-op (return False) without matplotlib
    _fixture(tmp_path)
    runs = load_runs(str(tmp_path))
    agg = aggregate_over_seeds(runs)
    hp = tmp_path / "horizon.png"
    pp = tmp_path / "poskl.png"
    assert plot_horizon_comparison(agg, str(hp)) is True
    assert plot_positional_kl(runs, str(pp)) is True
    assert hp.exists() and pp.exists()


def test_condition_key_groups_by_seed():
    # tag-based: strips _seedN so JSD betas / lambda variants stay distinct but seeds merge.
    assert condition_key({"_run_dir": "opd_rkl_seed0"}) == condition_key({"_run_dir": "opd_rkl_seed5"})
    assert condition_key({"_run_dir": "opd_jsd0.1_seed2"}) == "opd_jsd0.1"
    assert condition_key({"_run_dir": "opd_jsd0.5_seed0"}) != condition_key({"_run_dir": "opd_jsd0.1_seed0"})
