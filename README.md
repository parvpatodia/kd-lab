# kd-lab — on-policy distillation branch (scaffold)

Adds on-policy distillation to kd-lab and runs a controlled study that isolates the exposure-bias
mechanism. The method is GKD (Agarwal et al., 2023, arXiv:2306.13649), popularized by Thinking
Machines (2025). The contribution is the clean implementation of the divergence x data-source
design space inside kd-lab, the horizon-stratified mechanism study, and an honest reverse-KL
diversity-collapse failure-mode characterization. This is not a novelty claim.

## Read in this order
1. `START_HERE_PROMPT.md` — paste into a Claude Code session to begin.
2. `BUILDPLAN.md` — the runbook: motivation, prior art, hypotheses, experimental design, the
   8-phase loop with acceptance gates, fallback scopes, deliverables.
3. `DESIGN.md` — the technical spec: loss math, the file map (done vs SEAM), the condition matrix,
   metrics, and the Phase 0 integration checklist.

## Verified vs seam
Verified here against a numpy reference and brute-force enumeration: the divergences
(forward/reverse KL, generalized JSD), the next-token shift and response masking, the unbiased
pass@k estimator, and the pointer-chase task (determinism, gold-chain correctness, horizon dial,
error propagation). Every file byte-compiles.

Seams (depend on kd-lab and a GPU, marked `# SEAM` in code, listed in DESIGN.md section 9): model
and tokenizer loading, the sampler that returns rollouts with a correct response mask, the
off-policy dataset builder, the optimizer, the GSM8K loader, and the training step loop. The
control flow around them is written and correct; Phase 0 to 3 wires them in.

## Quick start (after Phase 0 wiring)
```bash
pip install torch numpy pyyaml pytest ruff mypy    # or the repo's [dev] extras
pytest -q                                          # divergence, task, metrics tests
python -m kd_lab.experiments.run --config configs/opd_rkl_smoke.yaml --dry-run
```

## Layout
```
kd_lab/distillation/   divergences.py, on_policy.py (distiller + rollout sources)
kd_lab/tasks/          pointer_chase.py (horizon-controllable synthetic task)
kd_lab/evaluation/     metrics.py, positional_kl.py (mechanism probe)
kd_lab/experiments/    run.py (config -> condition; --dry-run works)
configs/               opd_rkl_smoke.yaml
tests/                 test_scaffold.py (verified suite)
writeup/               medium_article_outline.md, linkedin.md
.github/workflows/     ci.yml      pyproject_additions.toml (merge into repo)
```

Note on the package name: the scaffold package is `kd_lab`. If the repo's package differs, rename
the directory and fix imports during Phase 0; the test imports use `kd_lab.` and a root
`conftest.py` puts the package on the path.
