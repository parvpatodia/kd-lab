# EXPERIMENTS — run log

Append one entry per run. Every number here must come from a logged run with a config hash.
No hand-edited or remembered numbers. A negative or null result is a valid entry. Pre-register
the primary metric (exact-match accuracy) before Phase 4 and do not change it after seeing data.

## How to log
For each run record: date, config file + hash, condition, seed(s), GPU-hours, the primary
metric with CI, and one line on anything surprising. Link the W&B run.

## Results table (fill as runs complete)

| date | condition | divergence | lambda | task | seeds | GPU-hrs | acc (mean) | 95% CI | pass@16 | notes |
|------|-----------|------------|--------|------|-------|---------|-----------|--------|---------|-------|
| _template_ | _OPD-RKL_ | _reverse_kl_ | _1.0_ | _pointer-chase_ | _0,1,2_ | _TBD_ | _TBD_ | _[lo, hi]_ | _TBD_ | _example row, replace with a real run_ |

## Horizon analysis (RQ1 / H1)
Record off-policy vs on-policy accuracy per horizon bucket once Phase 4 runs. The headline plot
is accuracy vs horizon for B-best (off-policy) and OPD-RKL (on-policy) with CIs, plus the
positional teacher-student KL curve. State plainly whether the gap widens with horizon.

## Ablations (RQ2 / RQ3)
- Divergence sweep (forward_kl, reverse_kl, generalized_jsd beta in {0.1, 0.5, 0.9}): accuracy and
  diversity (pass@k, entropy, distinct-n). Note the reverse-KL diversity-collapse result either way.
- Lambda sweep {0.0, 0.25, 0.5, 0.75, 1.0}: accuracy vs lambda; where do the gains saturate.

## Phase log (no-run phases recorded here; numbered runs go below)

### 2026-06-30  Phase 1 — Divergences + tests (no GPU run)
- Added `tests/test_phase1.py`: reverse-KL gradient sign on a 2-token vocab (autograd vs the
  closed form `dRKL/dz_i = q_i(log(q_i/p_i) - RKL)`); numpy closed-form reference cross-check for
  FKL/RKL/JSD; tau^2 scaling isolated from softening; masked positions contribute exactly zero to
  loss and gradient; CPU end-to-end `OnPolicyDistiller.step` on a 2-layer toy model (student
  updates, teacher stays frozen with no grad, loss falls over 30 steps).
- Infra: created `pyproject.toml` (installable package + single-source ruff/mypy/pytest config,
  tests-only E402 ignore for the `importorskip` pattern); fixed `F841` in the `run.py` SEAM;
  cleared pre-existing `E501`/`UP035`/`F401` in the scaffold; CI installs via `.[dev]` and mypy is
  now blocking.
- Gate: `pytest` 32 passed (26 scaffold + 6 new); `ruff` clean; `mypy` clean (11 files); dry-run
  intact. No training run, so no numbers produced (honesty clause).
- Note: the TRL `GKDTrainer` loss-parity oracle is deferred to Phase 3 (needs `trl` + a tiny model).

### 2026-06-30  Phase 0 — Orient (no GPU run)
- Read BUILDPLAN + DESIGN + the full scaffold. Floor verified: `pytest -q` = 26 passed (torch
  2.10.0 present, divergence tests ran); `run --config configs/opd_rkl_smoke.yaml --dry-run` prints
  the resolved plan (n_train=512, eval horizons 1..6 x100).
- Finding: no legacy off-policy KD exists to reuse (kd-lab is greenfield here). Baselines map onto
  the scaffold's own abstractions: B2 = OnPolicyDistiller + ForwardKL + TeacherRolloutSource
  (config-only); B0/B1 = a new small hard-label CE objective (gold / teacher-generated data).
- Finding: no saved Recall/knowledge cards on OPD/GKD/reverse KL/seq-level KD; design is from the
  cited papers and the scaffold.
- Blocker: git repo is rooted at /Users/parvpatodia (remote = Memories.ai-Hackathon-) and the
  scaffold is untracked; feature-branch/commit/PR workflow is blocked pending a human decision on
  re-init. No git changes made. DESIGN.md section 9 updated with the real integration points.

## Per-run entries
<!-- Newest first. Example shape:
### 2026-07-XX  OPD-RKL pointer-chase  (config hash abcdef0)
- seeds 0,1,2; 1.7 GPU-hrs total; vLLM sampling, bf16
- exact-match 0.XX [0.XX, 0.XX]; pass@16 0.XX
- surprise: gap vs B1 grows from k=2 to k=6 as predicted / not as predicted
-->
