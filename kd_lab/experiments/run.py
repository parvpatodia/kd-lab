"""Experiment runner: turn a YAML config into a single training + eval condition.

Condition routing (``distillation.method``):
  * ``sft``       B0: SupervisedDistiller on gold targets.
  * ``seq_kd``    B1: SupervisedDistiller on teacher generations (Kim & Rush, 2016).
  * ``logit_kd``  B2: OnPolicyDistiller + forward KL on gold data, teacher-scored (Hinton, 2015).
  * ``on_policy`` OPD: OnPolicyDistiller + the configured divergence on student rollouts
                  (StudentRolloutSource, or MixedRolloutSource when on_policy_fraction < 1).

What runs without a model/GPU: config parsing, divergence construction, pointer-chase data, the
resolved plan (``--dry-run``), and (via dependency injection) the whole ``run_condition`` loop and
eval on tiny CPU models. What needs the A100: ``load_models`` (real Qwen2.5 checkpoints) and the
scale at which the accuracy numbers are meaningful. ``transformers`` is imported lazily inside
``load_models`` so this module imports without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from ..distillation.divergences import Divergence, build_divergence
from ..distillation.on_policy import (
    MixedRolloutSource,
    OnPolicyDistiller,
    Rollouts,
    Sampler,
    StudentRolloutSource,
    TeacherRolloutSource,
)
from ..distillation.sampling import HFSampler, render_prompt
from ..distillation.supervised import SupervisedDistiller
from ..evaluation.metrics import (
    aggregate_pass_at_k,
    distinct_n,
    empirical_token_entropy,
    horizon_stratified_accuracy,
)
from ..evaluation.positional_kl import positional_teacher_student_kl
from ..tasks.pointer_chase import PointerChaseConfig, make_dataset, make_eval_sets, score_final

_SUPERVISED = ("sft", "seq_kd")
_OFF_POLICY = ("sft", "seq_kd", "logit_kd", "off_policy")


@dataclass
class ResolvedCondition:
    run_name: str
    method: str
    divergence_name: str
    lam: float
    seed: int


@dataclass
class RunComponents:
    """Everything ``run_condition`` needs, injectable so the loop is testable without HF/GPU."""

    student: object
    teacher: object
    tokenizer: object
    sampler: Sampler  # on-policy rollouts during training
    eval_sampler: Sampler  # greedy generation at eval
    off_dataset: object  # exposes .batch(examples) -> Rollouts
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def build_pointer_chase_data(cfg: dict):
    """Build train and per-horizon eval sets from the task block. Fully functional."""
    t = cfg["task"]
    pc = PointerChaseConfig(num_nodes=t.get("num_nodes", 32), shuffle_table=t.get("shuffle_table", True))
    train = make_dataset(n=t["n_train"], ks=t["train_k"], base_seed=cfg.get("seed", 0), cfg=pc)
    eval_sets = make_eval_sets(n_per_k=t["n_eval_per_k"], ks=t["eval_k"], cfg=pc)
    return train, eval_sets


def build_task_data(cfg: dict):
    """Dispatch on ``task.name``: pointer_chase (synthetic, offline) or gsm8k (external validity)."""
    name = cfg["task"].get("name", "pointer_chase")
    if name == "pointer_chase":
        return build_pointer_chase_data(cfg)
    if name == "gsm8k":
        from ..tasks.gsm8k import eval_sets_by_length, load_gsm8k  # lazy: needs `datasets`

        t = cfg["task"]
        train = load_gsm8k("train", n=t.get("n_train"))
        test = load_gsm8k("test", n=t.get("n_eval_total"))
        return train, eval_sets_by_length(test)
    raise ValueError(f"unknown task: {name}")


def get_scorer(cfg: dict):
    """Return the exact-match scorer for the configured task."""
    if cfg["task"].get("name", "pointer_chase") == "gsm8k":
        from ..tasks.gsm8k import score_gsm8k

        return score_gsm8k
    return score_final


def build_rollout_source(cfg: dict, student, sampler, off_dataset):
    """Select the data source from the distillation block and the method."""
    d = cfg["distillation"]
    method = d["method"]
    if method in _OFF_POLICY:
        return TeacherRolloutSource(off_dataset)
    lam = float(d.get("on_policy_fraction", 1.0))
    on_src = StudentRolloutSource(student, sampler)
    if lam >= 1.0:
        return on_src
    return MixedRolloutSource(on_src, TeacherRolloutSource(off_dataset), lam)


def build_distiller(cfg: dict, student, teacher, rollout_source, divergence: Divergence, optimizer):
    """Route method -> distiller. Supervised for B0/B1; teacher-scored for B2 and on-policy."""
    method = cfg["distillation"]["method"]
    device = cfg.get("device", "cuda")
    grad_clip = cfg["optim"].get("grad_clip", 1.0)
    if method in _SUPERVISED:
        return SupervisedDistiller(student, rollout_source, optimizer, device=device, grad_clip=grad_clip)
    return OnPolicyDistiller(
        student, teacher, rollout_source, divergence, optimizer, device=device, grad_clip=grad_clip
    )


# --------------------------------------------------------------------------------------
# Off-policy dataset: encode prompt + target into Rollouts (gold for B0/B2, teacher-gen for B1).
# --------------------------------------------------------------------------------------
def _encode_example(tokenizer, prompt: str, target: str, max_len: int, eos_id: int | None):
    # chat-template the prompt (Instruct models) so off-policy training matches how the student
    # is prompted at generation/eval time.
    p = tokenizer(render_prompt(tokenizer, prompt), add_special_tokens=False)["input_ids"]
    t = tokenizer(target, add_special_tokens=False)["input_ids"]
    if eos_id is not None:
        t = list(t) + [eos_id]
    ids = (list(p) + list(t))[:max_len]
    resp = ([0] * len(p) + [1] * len(t))[:max_len]
    return ids, resp


class EncodedRolloutDataset:
    """Off-policy dataset: encodes each example's prompt + target and pads a batch to Rollouts."""

    def __init__(self, tokenizer, *, max_len: int, eos_token_id: int | None, pad_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.eos_token_id = eos_token_id
        self.pad_token_id = int(pad_token_id)

    def batch(self, examples) -> Rollouts:
        enc = [
            _encode_example(self.tokenizer, ex["prompt"], ex["target"], self.max_len, self.eos_token_id)
            for ex in examples
        ]
        b = len(enc)
        maxlen = max((len(ids) for ids, _ in enc), default=1)
        input_ids = torch.full((b, maxlen), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, maxlen), dtype=torch.long)
        response_mask = torch.zeros((b, maxlen), dtype=torch.long)
        for i, (ids, resp) in enumerate(enc):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1
            response_mask[i, :length] = torch.tensor(resp, dtype=torch.long)
        return Rollouts(input_ids=input_ids, attention_mask=attention_mask, response_mask=response_mask)


def decode_responses(tokenizer, rollouts: Rollouts) -> list[str]:
    """Decode just the response tokens (response_mask == 1) of each row."""
    ids = rollouts.input_ids.cpu()
    mask = rollouts.response_mask.bool().cpu()
    out = []
    for row_ids, row_mask in zip(ids, mask, strict=True):
        out.append(tokenizer.decode(row_ids[row_mask].tolist()))
    return out


def response_token_ids(rollouts: Rollouts) -> list[list[int]]:
    """The response token ids per row (for distinct-n and token entropy)."""
    ids = rollouts.input_ids.cpu()
    mask = rollouts.response_mask.bool().cpu()
    return [row_ids[row_mask].tolist() for row_ids, row_mask in zip(ids, mask, strict=True)]


# ---- SEAMS: real HF wiring, exercised on the A100 (import transformers lazily) ----
def load_models(cfg: dict):
    """Load student (trainable) + teacher (frozen) + shared tokenizer. Needs a GPU for real sizes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = cfg["models"]
    dtype = getattr(torch, m.get("dtype", "bfloat16"))
    device = cfg.get("device", "cuda")
    tokenizer = AutoTokenizer.from_pretrained(m["student"])
    # `dtype=` is the current kwarg (transformers >=4.44 and 5.x); `torch_dtype` was removed in 5.x.
    student = AutoModelForCausalLM.from_pretrained(m["student"], dtype=dtype).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(m["teacher"], dtype=dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if student.config.vocab_size != teacher.config.vocab_size:
        raise ValueError(
            f"vocab mismatch: student {student.config.vocab_size} vs teacher {teacher.config.vocab_size}; "
            "the divergences assume an aligned vocab axis (keep the same tokenizer family)"
        )
    return student, teacher, tokenizer


def build_sampler(cfg: dict, tokenizer, *, greedy: bool = False) -> HFSampler:
    s = cfg["sampling"]
    return HFSampler(
        tokenizer,
        max_new_tokens=s.get("max_new_tokens", 64),
        temperature=0.0 if greedy else s.get("temperature", 1.0),
        top_p=s.get("top_p", 1.0),
        device=cfg.get("device", "cuda"),
    )


def build_offpolicy_dataset(cfg: dict, train_examples, tokenizer, teacher=None, sampler=None):
    """Gold targets for sft/logit_kd; teacher generations for seq_kd."""
    method = cfg["distillation"]["method"]
    examples = [dict(ex) for ex in train_examples]
    if method == "seq_kd":
        if teacher is None or sampler is None:
            raise ValueError("seq_kd needs the teacher and a sampler to generate targets")
        roll = sampler.generate(teacher, examples)
        gen = decode_responses(tokenizer, roll)
        for ex, g in zip(examples, gen, strict=True):
            ex["target"] = g
    return EncodedRolloutDataset(
        tokenizer,
        max_len=cfg["optim"].get("max_seq_len", 512),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0,
    )


def build_optimizer(cfg: dict, student):
    """AdamW + cosine schedule with warmup (pure torch; no transformers dependency)."""
    o = cfg["optim"]
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=o["lr"],
        weight_decay=o.get("weight_decay", 0.0),
    )
    total = int(o["max_steps"])
    warmup = int(o.get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# --------------------------------------------------------------------------------------
# Core loop + eval (injectable via RunComponents; no HF/GPU dependency in the control flow)
# --------------------------------------------------------------------------------------
def run_condition(cfg: dict, comp: RunComponents, *, results_root: str = "results") -> dict:
    """Train one condition then evaluate. Returns the metrics dict and writes it to disk."""
    seed_everything(int(cfg.get("seed", 0)))
    train_examples, eval_sets = build_task_data(cfg)

    divergence = build_divergence(
        cfg["distillation"]["divergence"],
        temperature=cfg["distillation"].get("temperature", 1.0),
        beta=cfg["distillation"].get("beta", 0.5),
    )
    rollout_source = build_rollout_source(cfg, comp.student, comp.sampler, comp.off_dataset)
    distiller = build_distiller(cfg, comp.student, comp.teacher, rollout_source, divergence, comp.optimizer)

    batch_size = int(cfg["optim"]["batch_size"])
    max_steps = int(cfg["optim"]["max_steps"])
    rng = random.Random(int(cfg.get("seed", 0)))
    train_log = []
    for step in range(max_steps):
        batch = [rng.choice(train_examples) for _ in range(batch_size)]
        metrics = distiller.step(batch)
        if comp.scheduler is not None:
            comp.scheduler.step()
        if step % max(1, max_steps // 10) == 0 or step == max_steps - 1:
            train_log.append({"step": step, "loss": metrics["loss"]})

    eval_out = evaluate(cfg, comp, eval_sets)

    results = {
        "plan": resolved_plan(cfg).__dict__,
        "config_hash": config_hash(cfg),
        "n_train": len(train_examples),
        "train_log": train_log,
    }
    results.update(eval_out)  # horizon_accuracy, positional_kl, and optional pass_at_k/diversity
    _save_results(cfg, results, results_root)
    return results


def evaluate(cfg: dict, comp: RunComponents, eval_sets: dict) -> dict:
    """Greedy horizon-stratified exact-match accuracy + the positional teacher-student KL probe."""
    scorer = get_scorer(cfg)
    eval_bs = int(cfg.get("eval", {}).get("eval_batch_size", 16))
    records: list[dict] = []
    for k, examples in eval_sets.items():
        for i in range(0, len(examples), eval_bs):  # chunk: generation OOMs on a full horizon set
            chunk = examples[i : i + eval_bs]
            texts = decode_responses(comp.tokenizer, comp.eval_sampler.generate(comp.student, chunk))
            records.extend(
                {"k": k, "correct": int(scorer(txt, ex))} for ex, txt in zip(chunk, texts, strict=True)
            )
    horizon = horizon_stratified_accuracy(records)

    # positional-KL probe on one mid-horizon eval set (RQ1 mechanism diagnostic).
    ks = sorted(eval_sets)
    probe_examples = eval_sets[ks[len(ks) // 2]][: int(cfg.get("eval", {}).get("probe_examples", 16))]
    probe_roll = comp.eval_sampler.generate(comp.student, probe_examples)
    pk = positional_teacher_student_kl(
        comp.student,
        comp.teacher,
        probe_roll,
        device=cfg.get("device", "cuda"),
        direction="reverse",
        batch_size=min(eval_bs, 4),  # the scoring forward is the heaviest op (two [B,T,V] tensors)
    )
    # keep only populated positions, as plain lists for JSON.
    valid = pk["count"] > 0
    positional = {
        "position": pk["position"][valid].tolist(),
        "mean_kl": [float(x) for x in pk["mean_kl"][valid]],
    }
    out = {"horizon_accuracy": horizon, "positional_kl": positional}
    out.update(_passk_and_diversity(cfg, comp, eval_sets))
    return out


def _passk_and_diversity(cfg: dict, comp: RunComponents, eval_sets: dict) -> dict:
    """Optional H2 metrics: pass@k plus diversity (distinct-n, token entropy) from sampled outputs.

    Gated by ``eval.n_samples_for_pass_at_k`` (0 disables). Uses the sampling sampler (temperature
    > 0), not the greedy eval sampler, over a capped pool of eval examples to bound cost.
    """
    ecfg = cfg.get("eval", {})
    n = int(ecfg.get("n_samples_for_pass_at_k", 0))
    if n <= 0:
        return {}
    ks = [k for k in ecfg.get("pass_at_k", [1, 4, 16]) if k <= n]
    cap = int(ecfg.get("passk_examples", 50))
    eval_bs = int(ecfg.get("eval_batch_size", 16))
    pool = [ex for exs in eval_sets.values() for ex in exs][:cap]
    scorer = get_scorer(cfg)

    correct = [0] * len(pool)
    gen_tokens: list[list[int]] = []
    for _ in range(n):
        for i in range(0, len(pool), eval_bs):  # chunk generation to bound memory
            chunk = pool[i : i + eval_bs]
            roll = comp.sampler.generate(comp.student, chunk)  # sampling (temperature > 0)
            texts = decode_responses(comp.tokenizer, roll)
            for j, (ex, txt) in enumerate(zip(chunk, texts, strict=True)):
                if scorer(txt, ex):
                    correct[i + j] += 1
            gen_tokens.extend(response_token_ids(roll))

    return {
        "pass_at_k": {f"pass@{k}": aggregate_pass_at_k(correct, n, k) for k in ks},
        "diversity": {
            "distinct_1": distinct_n(gen_tokens, 1),
            "distinct_2": distinct_n(gen_tokens, 2),
            "token_entropy": empirical_token_entropy(gen_tokens),
        },
    }


def _save_results(cfg: dict, results: dict, results_root: str) -> None:
    out_dir = Path(results_root) / cfg.get("run_name", "unnamed")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    _save_horizon_figure(results, out_dir / "horizon_accuracy.png")


def _save_horizon_figure(results: dict, path: Path) -> None:
    """Best-effort horizon accuracy figure. Skips silently if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ha = results["horizon_accuracy"]
    ks = sorted(ha, key=lambda x: int(x))
    means = [ha[k]["mean"] for k in ks]
    los = [ha[k]["mean"] - ha[k]["ci_low"] for k in ks]
    his = [ha[k]["ci_high"] - ha[k]["mean"] for k in ks]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar([int(k) for k in ks], means, yerr=[los, his], marker="o", capsize=3)
    ax.set_xlabel("horizon k")
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(results["plan"]["run_name"])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_and_eval(cfg: dict) -> dict:
    """Full condition on real models (A100 path). Builds components, then runs the loop + eval."""
    seed_everything(int(cfg.get("seed", 0)))
    train_examples, _ = build_task_data(cfg)
    student, teacher, tokenizer = load_models(cfg)
    sampler = build_sampler(cfg, tokenizer)
    eval_sampler = build_sampler(cfg, tokenizer, greedy=True)
    off_dataset = build_offpolicy_dataset(cfg, train_examples, tokenizer, teacher=teacher, sampler=sampler)
    optimizer, scheduler = build_optimizer(cfg, student)
    comp = RunComponents(
        student=student,
        teacher=teacher,
        tokenizer=tokenizer,
        sampler=sampler,
        eval_sampler=eval_sampler,
        off_dataset=off_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return run_condition(cfg, comp)


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
        train, eval_sets = build_task_data(cfg)
        build_divergence(
            cfg["distillation"]["divergence"],
            temperature=cfg["distillation"].get("temperature", 1.0),
            beta=cfg["distillation"].get("beta", 0.5),
        )
        print(
            json.dumps(
                {
                    "plan": plan.__dict__,
                    "config_hash": config_hash(cfg),
                    "n_train": len(train),
                    "eval_horizons": {k: len(v) for k, v in eval_sets.items()},
                    "divergence_ok": True,
                },
                indent=2,
            )
        )
        return
    train_and_eval(cfg)


if __name__ == "__main__":
    main()
