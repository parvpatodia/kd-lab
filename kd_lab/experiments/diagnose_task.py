"""Zero-shot task-difficulty probe: greedy exact-match accuracy of the teacher and the base
(untrained) student on the eval sets, by horizon, plus one sample generation each.

Why this exists: on-policy distillation can only lift the student toward the teacher. If the
teacher is itself near the floor on the task, there is no capability gap to distill and the
horizon study is floor-vs-floor (uninformative). This probe checks that before the sweep spends
compute. No training happens here.

Run: ``python -m kd_lab.experiments.diagnose_task --config configs/calib_opd_rkl.yaml --n 30``
"""

from __future__ import annotations

import argparse

from ..tasks.pointer_chase import PointerChaseConfig, make_eval_sets, score_final
from .run import build_sampler, decode_responses, load_config, load_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=30, help="examples per horizon")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t = cfg["task"]
    pc = PointerChaseConfig(num_nodes=t.get("num_nodes", 24), shuffle_table=t.get("shuffle_table", True))
    eval_sets = make_eval_sets(n_per_k=args.n, ks=t["eval_k"], cfg=pc)

    student, teacher, tokenizer = load_models(cfg)
    greedy = build_sampler(cfg, tokenizer, greedy=True)

    for name, model in [("teacher (1.5B)", teacher), ("student base (0.5B, untrained)", student)]:
        print(f"\n=== {name} ===")
        for k, examples in eval_sets.items():
            roll = greedy.generate(model, examples)
            texts = decode_responses(tokenizer, roll)
            acc = sum(score_final(x, e) for x, e in zip(texts, examples, strict=True)) / len(examples)
            print(f"  k={k}: greedy exact-match = {acc:.3f}  (n={len(examples)})")
        # one sample generation for eyeballing format/behavior
        ex = eval_sets[t["eval_k"][0]][0]
        sample = decode_responses(tokenizer, greedy.generate(model, [ex]))[0]
        print(f"  sample gold answer: {ex['answer']}")
        print(f"  sample generation: {sample[:300]!r}")


if __name__ == "__main__":
    main()
