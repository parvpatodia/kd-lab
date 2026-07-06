"""Aggregate per-run ``metrics.json`` into the headline table and figures.

Pure post-processing over the schema ``run_condition`` writes (no models): the RQ1 horizon plot
(accuracy vs horizon k per condition, averaged over seeds with a spread band), the positional
teacher-student KL probe, and a CSV results table with one row per run. Matplotlib is optional;
the table and the numeric aggregation work without it.

metrics.json schema (JSON turns the integer horizon keys into strings; we normalize back to int):
  plan{run_name,method,divergence_name,lam,seed}, config_hash, horizon_accuracy{k:{mean,ci_low,
  ci_high,n}}, positional_kl{position[],mean_kl[]}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

_SEED_RE = re.compile(r"_seed\d+$")


def load_runs(results_root: str) -> list[dict]:
    """Load every ``<run>/metrics.json`` under ``results_root``, newest schema tolerated."""
    runs = []
    for mj in sorted(Path(results_root).glob("*/metrics.json")):
        data = json.loads(mj.read_text())
        data["_run_dir"] = mj.parent.name
        # normalize horizon keys to int
        data["horizon_accuracy"] = {int(k): v for k, v in data.get("horizon_accuracy", {}).items()}
        runs.append(data)
    return runs


def condition_key(run: dict) -> str:
    """Group runs that differ only by seed, using the sweep tag (the run dir minus ``_seedN``).

    The tag distinguishes conditions the metrics plan cannot: JSD(0.1) vs JSD(0.5) vs JSD(0.9)
    (beta is not stored in the plan) and each lambda in the reverse-KL sweep.
    """
    return _SEED_RE.sub("", run["_run_dir"])


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def aggregate_over_seeds(runs: list[dict]) -> dict:
    """Return ``{condition_key: {horizon: {mean, std, n_seeds}}}`` averaging each run's per-horizon
    accuracy across seeds."""
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        key = condition_key(run)
        for k, stats in run["horizon_accuracy"].items():
            grouped[key][k].append(stats["mean"])
    out: dict = {}
    for key, per_k in grouped.items():
        out[key] = {}
        for k in sorted(per_k):
            mean, std = _mean_std(per_k[k])
            out[key][k] = {"mean": mean, "std": std, "n_seeds": len(per_k[k])}
    return out


def write_results_table(runs: list[dict], path: str) -> None:
    """One row per run: plan fields, config hash, and per-horizon accuracy means."""
    horizons = sorted({k for run in runs for k in run["horizon_accuracy"]})
    fields = ["run_dir", "method", "divergence", "lambda", "seed", "config_hash"] + [
        f"acc_k{k}" for k in horizons
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for run in runs:
            p = run["plan"]
            row = [
                run["_run_dir"],
                p["method"],
                p["divergence_name"],
                p["lam"],
                p["seed"],
                run.get("config_hash", ""),
            ]
            row += [run["horizon_accuracy"].get(k, {}).get("mean", "") for k in horizons]
            w.writerow(row)


def _label(key: str) -> str:
    return key  # the condition tag is already the readable label


def plot_horizon_comparison(aggregated: dict, path: str, keys: list[tuple] | None = None) -> bool:
    """RQ1 headline: accuracy vs horizon k per condition (mean over seeds, +/- std band).

    Returns True if a figure was written, False if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    keys = keys if keys is not None else list(aggregated)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in keys:
        per_k = aggregated[key]
        ks = sorted(per_k)
        means = [per_k[k]["mean"] for k in ks]
        stds = [per_k[k]["std"] for k in ks]
        ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3, label=_label(key))
    ax.set_xlabel("horizon k")
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Accuracy vs horizon (mean over seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def plot_positional_kl(runs: list[dict], path: str, keys: list[tuple] | None = None) -> bool:
    """Positional teacher-student KL vs token index (one curve per selected condition, seed 0)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for run in runs:
        if run["plan"]["seed"] != 0:
            continue
        if keys is not None and condition_key(run) not in keys:
            continue
        pk = run.get("positional_kl", {})
        if pk.get("position"):
            ax.plot(pk["position"], pk["mean_kl"], marker=".", label=_label(condition_key(run)))
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("response token position")
    ax.set_ylabel("teacher-student KL")
    ax.set_title("Positional KL over student rollouts (seed 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="root holding <run>/metrics.json dirs")
    ap.add_argument("--out", default="results/analysis", help="output dir for table + figures")
    args = ap.parse_args()

    runs = [r for r in load_runs(args.results) if r["_run_dir"] not in ("smoke_cluster", "calib_opd_rkl")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_results_table(runs, str(out / "results_table.csv"))
    agg = aggregate_over_seeds(runs)
    fig_ok = plot_horizon_comparison(agg, str(out / "horizon_accuracy.png"))
    plot_positional_kl(runs, str(out / "positional_kl.png"))

    print(f"aggregated {len(runs)} runs -> {out} (figures: {'yes' if fig_ok else 'matplotlib missing'})")
    for key in sorted(agg):
        cells = "  ".join(f"k{k}={agg[key][k]['mean']:.2f}" for k in sorted(agg[key]))
        print(f"  {_label(key):24s} (n={next(iter(agg[key].values()))['n_seeds']})  {cells}")


if __name__ == "__main__":
    main()
