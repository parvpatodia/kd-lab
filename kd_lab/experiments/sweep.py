"""Expand a base config into the BUILDPLAN condition matrix (method x divergence x lambda x seed).

Pure config manipulation, no models. Writes one resolved YAML per condition (DESIGN section 7:
"one file per condition; seeded") that ``experiments/run.py`` consumes, and a manifest listing
them in order. The SLURM array script (``experiments/slurm_sweep.sh``) launches one array task per
generated config.

Conditions (BUILDPLAN section 5.3):
  baselines B0 sft, B1 seq_kd, B2 logit_kd (forward KL on gold, teacher-scored);
  on-policy divergence sweep OPD-RKL / OPD-FKL / OPD-JSD(beta in {0.1,0.5,0.9}) at lambda=1;
  lambda sweep OPD-RKL at lambda in {0.0,0.25,0.5,0.75} (1.0 is OPD-RKL above).
"""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import yaml

from .run import load_config


def default_conditions() -> list[dict]:
    """The condition matrix as distillation-block overrides, each with a short ``tag``."""
    conds: list[dict] = [
        {"tag": "b0_sft", "method": "sft", "divergence": "forward_kl", "on_policy_fraction": 0.0},
        {"tag": "b1_seqkd", "method": "seq_kd", "divergence": "forward_kl", "on_policy_fraction": 0.0},
        {"tag": "b2_logitkd", "method": "logit_kd", "divergence": "forward_kl", "on_policy_fraction": 0.0},
        {"tag": "opd_rkl", "method": "on_policy", "divergence": "reverse_kl", "on_policy_fraction": 1.0},
        {"tag": "opd_fkl", "method": "on_policy", "divergence": "forward_kl", "on_policy_fraction": 1.0},
    ]
    for beta in (0.1, 0.5, 0.9):
        conds.append(
            {
                "tag": f"opd_jsd{beta}",
                "method": "on_policy",
                "divergence": "generalized_jsd",
                "beta": beta,
                "on_policy_fraction": 1.0,
            }
        )
    for lam in (0.0, 0.25, 0.5, 0.75):
        conds.append(
            {
                "tag": f"opd_rkl_lam{lam}",
                "method": "on_policy",
                "divergence": "reverse_kl",
                "on_policy_fraction": lam,
            }
        )
    return conds


def expand(base: dict, conditions: list[dict], seeds: list[int]) -> list[dict]:
    """Cross conditions with seeds into resolved configs with unique run names."""
    out: list[dict] = []
    for cond, seed in itertools.product(conditions, seeds):
        cfg = copy.deepcopy(base)
        d = cfg.setdefault("distillation", {})
        for key in ("method", "divergence", "on_policy_fraction", "beta"):
            if key in cond:
                d[key] = cond[key]
        cfg["seed"] = int(seed)
        cfg["run_name"] = f"{cond['tag']}_seed{seed}"
        out.append(cfg)
    return out


def write_configs(configs: list[dict], out_dir: str) -> list[str]:
    """Write each config to ``out_dir/<run_name>.yaml`` and a manifest; return the paths in order."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for cfg in configs:
        p = out / f"{cfg['run_name']}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(str(p))
    (out / "manifest.txt").write_text("\n".join(paths) + "\n")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base config YAML (full-scale, real models)")
    ap.add_argument("--out", default="configs/generated", help="output directory for per-condition configs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dry-run", action="store_true", help="print the plan without writing files")
    args = ap.parse_args()

    base = load_config(args.base)
    configs = expand(base, default_conditions(), args.seeds)
    if args.dry_run:
        for cfg in configs:
            d = cfg["distillation"]
            print(f"{cfg['run_name']:24s} method={d['method']:9s} div={d['divergence']:15s} "
                  f"lambda={d.get('on_policy_fraction')} seed={cfg['seed']}")
        print(f"\n{len(configs)} configs "
              f"({len(default_conditions())} conditions x {len(args.seeds)} seeds)")
        return
    paths = write_configs(configs, args.out)
    print(f"wrote {len(paths)} configs to {args.out} (SLURM array 0-{len(paths) - 1})")


if __name__ == "__main__":
    main()
