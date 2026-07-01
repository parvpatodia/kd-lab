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

### 2026-06-30  Phase 3 — On-policy distiller + TRL oracle (no GPU run)
- The on-policy distiller was implemented in Phase 1 (CPU smoke: loss decreases over 30 steps,
  student updates, teacher stays frozen with no grad). Phase 3 adds the correctness oracle.
- `tests/test_trl_oracle.py`: hand-rolled divergences vs TRL `GKDTrainer.generalized_jsd_loss`
  (trl 1.7.0). Exact parity: forward KL == TRL beta=0 (max|diff| 0.0), reverse KL == TRL beta=1
  (0.0), generalized JSD == TRL for beta in {0.1,0.3,0.5,0.9} (~1e-8). Endpoint convention
  difference documented and pinned: our JSD(0)=JSD(1)=0 (true JSD) vs TRL's special-cased raw KL
  (DESIGN section 2). This is stronger than a single-config numeric check.
- Launch infra (Phase 4 prep, no GPU): `kd_lab/experiments/sweep.py` (12-condition x 3-seed = 36
  config matrix; dry-run verified), `configs/pointer_chase_base.yaml` (full-scale starting point),
  `kd_lab/experiments/slurm_sweep.sh` (templated A100 array launcher).
- Gate: substance met (exact TRL loss-math parity; OPD loop trains and loss decreases at CPU toy
  scale). The real-model smoke-config training run is GPU-deferred. pytest 57 passed; ruff clean;
  mypy clean (14 files).
- BOUNDARY: Phases 4-6 (the experiments and their numbers) require the A100 cluster; no CUDA/SLURM
  here. Stopping for the cluster checkpoint (BUILDPLAN section 0). No accuracy numbers produced.

### 2026-06-30  Phase 2 — Off-policy baselines wired (no GPU run)
- Implemented the `run.py` SEAMs: `load_models` (transformers, lazy import + vocab-parity guard),
  `build_sampler` (HFSampler, greedy/sampled), `build_offpolicy_dataset` (`EncodedRolloutDataset`;
  gold for sft/logit_kd, teacher generations for seq_kd), `build_optimizer` (AdamW + cosine
  warmup), method routing (sft/seq_kd -> SupervisedDistiller; logit_kd/on_policy ->
  OnPolicyDistiller), and `run_condition` (train loop + greedy horizon eval + positional-KL probe
  + metrics.json + horizon figure).
- Validated offline (no HF, no GPU) via dependency injection: tiny torch models + a stub tokenizer
  + a tiled sampler drive the full sft and on_policy paths end to end (`tests/test_runner.py`).
- Gate status: the BUILDPLAN Phase 2 gate ("baselines run end-to-end on the A100, plausible
  accuracy") needs the cluster and is NOT met here (dev box has no CUDA/SLURM; HF downloads
  unavailable). The code is cluster-ready and the orchestration is verified offline. No accuracy
  numbers produced (honesty clause).
- Env note: local numpy 2.x caused a transformers/tokenizers ABI break; installing matplotlib
  pulled numpy back to 1.26.4 and cleared it. HF model *downloads* still fail here (egress), so
  real-model validation is deferred to the cluster.
- pytest 49 passed; ruff clean; mypy clean (13 files).

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
