"""Experiment runner: turn a YAML config into a single training+eval condition.

What works without a model: config parsing, divergence construction, pointer-chase dataset
and eval-set construction, and printing the resolved plan (``--dry-run``). What is a SEAM:
loading the student/teacher and tokenizer, the concrete ``Sampler``, the GSM8K loader, and
the per-step training loop, all of which depend on kd-lab's existing model/data utilities.

Phase 0 task for the agent: wire the SEAM functions to kd-lab's loaders, then remove the
NotImplementedError guards. The condition matrix (divergence x lambda x seed) is expanded by
``experiments/sweep`` configs that point at this runner.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import yaml

from ..distillation.divergences import build_divergence
from ..distillation.on_policy import (
    MixedRolloutSource,
    OnPolicyDistiller,
    StudentRolloutSource,
    TeacherRolloutSource,
)
from ..tasks.pointer_chase import PointerChaseConfig, make_dataset, make_eval_sets


@dataclass
class ResolvedCondition:
    run_name: str
    method: str
    divergence_name: str
    lam: float
    seed: int


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pointer_chase_data(cfg: dict):
    """Build train and per-horizon eval sets from the task block. Fully functional."""
    t = cfg["task"]
    pc = PointerChaseConfig(num_nodes=t.get("num_nodes", 32), shuffle_table=t.get("shuffle_table", True))
    train = make_dataset(n=t["n_train"], ks=t["train_k"], base_seed=cfg.get("seed", 0), cfg=pc)
    eval_sets = make_eval_sets(n_per_k=t["n_eval_per_k"], ks=t["eval_k"], cfg=pc)
    return train, eval_sets


def build_rollout_source(cfg: dict, student, sampler, off_dataset):
    """Select the data source from the distillation block.

    method=on_policy + lambda<1 -> MixedRolloutSource; lambda==1 -> StudentRolloutSource;
    method=off_policy -> TeacherRolloutSource (the baseline).
    """
    d = cfg["distillation"]
    on_src = StudentRolloutSource(student, sampler)
    off_src = TeacherRolloutSource(off_dataset)
    if d["method"] == "off_policy":
        return off_src
    lam = float(d.get("on_policy_fraction", 1.0))
    if lam >= 1.0:
        return on_src
    return MixedRolloutSource(on_src, off_src, lam)


# ---- SEAMS: wire these to kd-lab's existing utilities in Phase 0 ----
def load_models(cfg: dict):
    raise NotImplementedError("SEAM: load student/teacher (+ tokenizer) via kd-lab's model utils")


def build_sampler(cfg: dict):
    raise NotImplementedError("SEAM: return a Sampler (HF generate or vLLM) producing Rollouts")


def build_offpolicy_dataset(cfg: dict, train_examples, tokenizer):
    raise NotImplementedError("SEAM: precompute teacher/gold rollouts as an off-policy dataset")


def build_optimizer(cfg: dict, student):
    raise NotImplementedError("SEAM: AdamW + scheduler from cfg['optim'] over student params")


def train_and_eval(cfg: dict) -> dict:
    """Full condition. Calls the SEAM loaders, then runs the OnPolicyDistiller loop."""
    train_examples, eval_sets = build_pointer_chase_data(cfg)
    student, teacher, tokenizer = load_models(cfg)
    sampler = build_sampler(cfg)
    off_dataset = build_offpolicy_dataset(cfg, train_examples, tokenizer)
    optimizer = build_optimizer(cfg, student)
    rollout_source = build_rollout_source(cfg, student, sampler, off_dataset)
    divergence = build_divergence(
        cfg["distillation"]["divergence"],
        temperature=cfg["distillation"].get("temperature", 1.0),
        beta=cfg["distillation"].get("beta", 0.5),
    )
    distiller = OnPolicyDistiller(
        student, teacher, rollout_source, divergence, optimizer,
        device=cfg.get("device", "cuda"), grad_clip=cfg["optim"].get("grad_clip", 1.0),
    )
    # SEAM: batching of `train_examples` into prompts, the step loop to cfg['optim']['max_steps'],
    # logging, checkpointing, and the horizon/diversity/positional-KL evaluation on `eval_sets`.
    raise NotImplementedError("SEAM: implement the step loop + evaluation; see DESIGN.md Phase 3-5")


def resolved_plan(cfg: dict) -> ResolvedCondition:
    d = cfg["distillation"]
    return ResolvedCondition(
        run_name=cfg.get("run_name", "unnamed"),
        method=d["method"],
        divergence_name=d["divergence"],
        lam=float(d.get("on_policy_fraction", 1.0)),
        seed=int(cfg.get("seed", 0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate config + build data, no models")
    args = ap.parse_args()

    cfg = load_config(args.config)
    plan = resolved_plan(cfg)
    if args.dry_run:
        train, eval_sets = build_pointer_chase_data(cfg)
        # construct the divergence to validate the name/params early
        build_divergence(cfg["distillation"]["divergence"],
                          temperature=cfg["distillation"].get("temperature", 1.0),
                          beta=cfg["distillation"].get("beta", 0.5))
        print(json.dumps({
            "plan": plan.__dict__,
            "n_train": len(train),
            "eval_horizons": {k: len(v) for k, v in eval_sets.items()},
            "divergence_ok": True,
        }, indent=2))
        return
    train_and_eval(cfg)


if __name__ == "__main__":
    main()
