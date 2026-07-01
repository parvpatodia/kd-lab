"""Pointer-chase: a horizon-controllable synthetic task that isolates exposure bias.

Each example gives a random bijection f over N nodes (a permutation, so chains never dead-end)
as an in-context table, a start node, and a hop count k. The answer is f applied k times to
the start. The model must read the specific table (labels are arbitrary, so there is no
shortcut) and chase pointers k times, emitting each hop.

Why this task carries RQ1 cleanly:
  * Horizon = k is a single dial; reasoning length scales linearly with k.
  * Errors compound deterministically: a wrong hop sends the chain to a wrong node and every
    downstream hop is then wrong. This is exactly the exposure-bias setting on-policy
    distillation is supposed to fix. The set of intermediate nodes the student visits at
    inference differs from the teacher's golden chain, so off-policy training under-covers
    the recovery states while on-policy training practices them.
  * Exact-match scorable on the final node, with per-hop accuracy available for analysis.

Verified in tests/test_scaffold.py: determinism, the gold chain follows the table, the
verifier round-trips the gold target, the horizon dial, and downstream error propagation.

Pure standard library; no model dependency.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass
class PointerChaseConfig:
    num_nodes: int = 32
    shuffle_table: bool = True  # render table lines in random order so position is uninformative


def make_permutation(num_nodes: int, rng: random.Random) -> list[int]:
    """Return ``perm`` with ``perm[i] = f(i)``, a random bijection over ``range(num_nodes)``."""
    perm = list(range(num_nodes))
    rng.shuffle(perm)
    return perm


def chase(perm: list[int], start: int, k: int) -> list[int]:
    """Return the chain ``[start, f(start), ..., f^k(start)]`` of length ``k + 1``."""
    chain = [start]
    cur = start
    for _ in range(k):
        cur = perm[cur]
        chain.append(cur)
    return chain


def render_example(perm: list[int], start: int, k: int, rng: random.Random, cfg: PointerChaseConfig) -> dict:
    """Build one ``{prompt, target, answer, chain, k}`` example."""
    idx = list(range(cfg.num_nodes))
    if cfg.shuffle_table:
        rng.shuffle(idx)
    table = "\n".join(f"{i} -> {perm[i]}" for i in idx)
    prompt = (
        "Mapping (each line 'a -> b'):\n"
        f"{table}\n"
        f"Start: {start}\n"
        f"Follow the mapping {k} times. Show each step, then 'Final: <answer>'."
    )
    chain = chase(perm, start, k)
    target = " -> ".join(str(x) for x in chain) + f"\nFinal: {chain[-1]}"
    return {"prompt": prompt, "target": target, "answer": chain[-1], "chain": chain, "k": k}


def make_example(seed: int, k: int, cfg: PointerChaseConfig | None = None) -> dict:
    """Deterministic single example from ``seed`` and horizon ``k``."""
    cfg = cfg or PointerChaseConfig()
    rng = random.Random(seed)
    perm = make_permutation(cfg.num_nodes, rng)
    start = rng.randrange(cfg.num_nodes)
    return render_example(perm, start, k, rng, cfg)


def make_dataset(
    n: int, ks: list[int], base_seed: int = 0, cfg: PointerChaseConfig | None = None
) -> list[dict]:
    """Return ``n`` examples with horizons drawn uniformly from ``ks`` (reproducible)."""
    cfg = cfg or PointerChaseConfig()
    rng = random.Random(base_seed)
    out = []
    for i in range(n):
        k = rng.choice(ks)
        out.append(make_example(base_seed * 1_000_003 + i, k, cfg))
    return out


def make_eval_sets(
    n_per_k: int, ks: list[int], base_seed: int = 10_000, cfg: PointerChaseConfig | None = None
) -> dict:
    """Return ``{k: [examples]}`` with a fixed count per horizon, for the RQ1 horizon sweep."""
    cfg = cfg or PointerChaseConfig()
    sets: dict[int, list[dict]] = {}
    for k in ks:
        sets[k] = [make_example(base_seed + k * 100_000 + i, k, cfg) for i in range(n_per_k)]
    return sets


_FINAL_RE = re.compile(r"Final:\s*(-?\d+)")
_CHAIN_RE = re.compile(r"(?:-?\d+)(?:\s*->\s*-?\d+)+")
_INT_RE = re.compile(r"-?\d+")


def parse_final_answer(text: str) -> int | None:
    """Extract the model's final answer: prefer 'Final: <n>', else the last integer."""
    m = _FINAL_RE.findall(text)
    if m:
        return int(m[-1])
    nums = _INT_RE.findall(text)
    return int(nums[-1]) if nums else None


def parse_chain(text: str) -> list[int] | None:
    """Extract the last arrow-joined run of integers as the model's hop chain."""
    runs = _CHAIN_RE.findall(text)
    if not runs:
        return None
    return [int(x) for x in _INT_RE.findall(runs[-1])]


def score_final(prediction_text: str, example: dict) -> bool:
    """Exact match on the final node."""
    return parse_final_answer(prediction_text) == example["answer"]


def score_per_hop(prediction_text: str, example: dict) -> float:
    """Fraction of hops the model got right, by position (diagnostic for compounding error)."""
    pred = parse_chain(prediction_text)
    gold = example["chain"]
    if not pred:
        return 0.0
    correct = sum(1 for i in range(1, len(gold)) if i < len(pred) and pred[i] == gold[i])
    return correct / max(len(gold) - 1, 1)
