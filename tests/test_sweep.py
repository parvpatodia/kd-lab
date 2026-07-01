"""Tests for the condition-matrix sweep generator. Pure config, no models."""

from __future__ import annotations

from pathlib import Path

from kd_lab.experiments.sweep import default_conditions, expand, write_configs


def _base() -> dict:
    return {
        "distillation": {"method": "on_policy", "divergence": "reverse_kl"},
        "seed": 0,
        "run_name": "base",
    }


def test_expand_counts_and_unique_names():
    conds, seeds = default_conditions(), [0, 1, 2]
    cfgs = expand(_base(), conds, seeds)
    assert len(cfgs) == len(conds) * len(seeds)
    names = [c["run_name"] for c in cfgs]
    assert len(set(names)) == len(names)


def test_base_not_mutated_and_fields_set():
    base = _base()
    cfgs = expand(base, default_conditions(), [7])
    assert base["seed"] == 0 and base["run_name"] == "base"  # deep-copied, untouched
    for c in cfgs:
        assert c["seed"] == 7
        assert "method" in c["distillation"] and "divergence" in c["distillation"]


def test_beta_present_only_for_jsd():
    cfgs = expand(_base(), default_conditions(), [0])
    for c in cfgs:
        if c["distillation"]["divergence"] == "generalized_jsd":
            assert "beta" in c["distillation"]


def test_write_configs(tmp_path):
    cfgs = expand(_base(), default_conditions()[:3], [0])
    paths = write_configs(cfgs, str(tmp_path))
    assert len(paths) == 3
    assert all(Path(p).exists() for p in paths)
    assert (tmp_path / "manifest.txt").exists()
