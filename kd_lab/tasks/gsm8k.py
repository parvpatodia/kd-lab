"""GSM8K loader + scoring for external validity (RQ1-RQ3 secondary task).

Examples share the pointer-chase shape ({prompt, target, answer, k, ...}) so the runner reuses the
same pipeline. The "horizon" for GSM8K is the solution length in reasoning steps, bucketed so it
stratifies accuracy the way k stratifies the synthetic task (BUILDPLAN section 5.2). The dataset
download (``datasets.load_dataset``) runs on the cluster; the parsing, bucketing, and scoring
helpers are pure and tested offline.
"""

from __future__ import annotations

import re

_PROMPT_TMPL = (
    "Solve the math problem. Show your reasoning step by step, then end with a line "
    "'Final: <answer>'.\n\nProblem: {q}\n"
)
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+)")
_FINAL_RE = re.compile(r"Final:\s*(-?[\d,]+)")
_INT_RE = re.compile(r"-?\d[\d,]*")

# default step-count bucket edges: [<=2], [3-4], [5-6], [7+]
_DEFAULT_EDGES = (2, 4, 6)


def extract_gold_answer(answer_field: str) -> int:
    """GSM8K gold answers end with '#### <n>'. Return the integer (commas stripped)."""
    m = _GOLD_RE.search(answer_field)
    if not m:
        raise ValueError(f"no '#### <answer>' found in: {answer_field[:80]!r}")
    return int(m.group(1).replace(",", ""))


def num_reasoning_steps(answer_field: str) -> int:
    """Count non-empty reasoning lines before the '####' marker."""
    body = answer_field.split("####")[0].strip()
    return sum(1 for ln in body.splitlines() if ln.strip())


def length_bucket(n_steps: int, edges: tuple[int, ...] = _DEFAULT_EDGES) -> int:
    """Map a step count to a bucket index (0..len(edges)); the bucket is the GSM8K horizon."""
    for i, e in enumerate(edges):
        if n_steps <= e:
            return i
    return len(edges)


def make_example(question: str, answer_field: str, edges: tuple[int, ...] = _DEFAULT_EDGES) -> dict:
    """Build one {prompt, target, answer, n_steps, k} example from a GSM8K row."""
    gold = extract_gold_answer(answer_field)
    cot = answer_field.split("####")[0].strip()
    n_steps = num_reasoning_steps(answer_field)
    return {
        "prompt": _PROMPT_TMPL.format(q=question),
        "target": f"{cot}\nFinal: {gold}",
        "answer": gold,
        "n_steps": n_steps,
        "k": length_bucket(n_steps, edges),
    }


def parse_pred_answer(text: str) -> int | None:
    """Prefer 'Final: <n>', else the last integer. Commas stripped. Matches the task scorer style."""
    m = _FINAL_RE.findall(text)
    if m:
        return int(m[-1].replace(",", ""))
    nums = _INT_RE.findall(text)
    return int(nums[-1].replace(",", "")) if nums else None


def score_gsm8k(prediction_text: str, example: dict) -> bool:
    """Exact match on the final integer answer."""
    return parse_pred_answer(prediction_text) == example["answer"]


def load_gsm8k(
    split: str = "test", n: int | None = None, edges: tuple[int, ...] = _DEFAULT_EDGES
) -> list[dict]:
    """Load GSM8K (main config) into example dicts. Needs ``datasets`` (cluster-side, lazy import)."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    out = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        out.append(make_example(row["question"], row["answer"], edges))
    return out


def eval_sets_by_length(examples: list[dict]) -> dict:
    """Group examples into {length_bucket: [examples]} for the horizon-stratified eval."""
    sets: dict[int, list[dict]] = {}
    for ex in examples:
        sets.setdefault(ex["k"], []).append(ex)
    return {k: sets[k] for k in sorted(sets)}
