# EXPERIMENTS — run log

Append one entry per run. Every number here must come from a logged run with a config hash.
No hand-edited or remembered numbers. A negative or null result is a valid entry. Pre-register
the primary metric (exact-match accuracy) before Phase 4 and do not change it after seeing data.

## How to log
For each run record: date, config file + hash, condition, seed(s), GPU-hours, the primary
metric with CI, and one line on anything surprising. Link the W&B run.

## Results (pointer-chase, Qwen2.5-0.5B <- 1.5B, 400 steps, 3 seeds; base config hash a1dd895faf1f)

Task: pointer-chase, num_nodes=32, train horizons k in {2,3,4}, eval k in {2..8} (so **k>=5 is
extrapolation beyond the training horizon**). Primary metric: greedy exact-match accuracy. Values
are the mean over seeds 0,1,2 (per-run 95% bootstrap CIs over 100 eval examples are in each
`results/<run>/metrics.json`). Full table: `results/analysis/results_table.csv`. 36/36 runs, all
completed on Explorer V100-sxm2, ~1.3-1.5 GPU-hr each (~50 GPU-hr total). Sweep array 8088396,
8118524, 8143267, 8184440, 8185806.

| condition (method / divergence, lambda) | k2 | k3 | k4 | k5 | k6 | k7 | k8 |
|---|---|---|---|---|---|---|---|
| B0 SFT (gold, off-policy)          | 1.00 | 1.00 | 0.99 | 0.25 | 0.06 | 0.03 | 0.04 |
| B1 seq-KD (teacher-gen, off-policy)| 1.00 | 1.00 | 0.99 | 0.25 | 0.06 | 0.03 | 0.04 |
| B2 logit-KD (fwd-KL on gold, off)  | 0.07 | 0.17 | 0.09 | 0.09 | 0.11 | 0.13 | 0.12 |
| OPD-FKL (on-policy, lambda=1)      | 0.34 | 0.35 | 0.14 | 0.15 | 0.05 | 0.11 | 0.12 |
| OPD-JSD(0.1) (on-policy)           | 0.37 | 0.31 | 0.17 | 0.11 | 0.10 | 0.13 | 0.11 |
| OPD-JSD(0.5) (on-policy)           | 0.44 | 0.41 | 0.21 | 0.22 | 0.16 | 0.18 | 0.20 |
| OPD-JSD(0.9) (on-policy)           | 0.40 | 0.46 | 0.28 | 0.23 | 0.17 | 0.17 | 0.15 |
| **OPD-RKL (on-policy, lambda=1)**  | 0.48 | 0.49 | 0.31 | 0.22 | 0.19 | 0.23 | 0.21 |
| OPD-RKL lambda=0.0 (off-policy RKL)| 0.03 | 0.15 | 0.11 | 0.15 | 0.16 | 0.18 | 0.18 |
| OPD-RKL lambda=0.25               | 0.32 | 0.29 | 0.19 | 0.15 | 0.17 | 0.16 | 0.13 |
| OPD-RKL lambda=0.5                | 0.39 | 0.39 | 0.27 | 0.27 | 0.28 | 0.26 | 0.23 |
| OPD-RKL lambda=0.75              | 0.51 | 0.48 | 0.33 | 0.30 | 0.26 | 0.25 | 0.26 |

Figures: `results/figures/fig1_h1_offpolicy_vs_onpolicy.png`, `fig2_lambda_sweep.png`,
`fig3_divergence.png`, `fig4_positional_kl.png`.

## Hypothesis verdicts (pre-registered in BUILDPLAN section 3)

**H1 (exposure bias / horizon) — CONFIRMED, with a crossover.** Off-policy SFT/seq-KD are near
perfect in-distribution (k<=4 ~ 1.00) then collapse on extrapolation (k8 = 0.04). On-policy OPD-RKL
is worse in-distribution (k2 = 0.48) but degrades far more gently and dominates on extrapolation
(k8 = 0.21 vs 0.04, a ~5x gap; the curves cross near k=5). So the off-policy-vs-on-policy gap
depends on horizon and flips sign: off-policy wins where it has gold supervision, on-policy wins
where the student must handle states beyond the training horizon. Mechanism confirmed directly by
the positional probe: the SFT student's reverse-KL to the teacher on its own rollouts is large and
erratic (pos0 = 4.75, then ~0.5-1.6), while the on-policy student stays uniformly low and flat
(~0.0-0.17). The on-policy student trained on its own state distribution, so it does not drift.

**H2 (divergence) — accuracy part CONFIRMED; diversity part NOT TESTED.** For on-policy (lambda=1),
reverse KL > JSD > forward KL on exact-match (RKL k2=0.48 vs FKL k2=0.34; FKL collapses on the tail
like off-policy). Generalized JSD interpolates, best around beta=0.5-0.9. The diversity-collapse
claim (pass@k, entropy, distinct-n) was NOT tested: the sweep ran greedy eval only
(n_samples_for_pass_at_k=0). Testing it needs sampled-eval runs; the metrics are implemented and
tested but were descoped from this sweep to save compute. Recorded as open.

**H3 (data source / lambda) — PARTIALLY CONFIRMED.** Accuracy rises steeply with the on-policy
fraction and most of the gain arrives by lambda=0.5 (k2: 0.03 -> 0.32 -> 0.39 for lambda 0 ->
0.25 -> 0.5). But it is NOT strictly monotonic to lambda=1: mixed lambda=0.5-0.75 matches or beats
pure on-policy on the extrapolation tail (k6: 0.28 at lambda=0.5 vs 0.19 at lambda=1.0), and the
low-horizon peak is at lambda=0.75 (0.51) not 1.0 (0.48). So "more on-policy is better, with
diminishing returns, mostly saturated by lambda~0.5-0.75" holds; strict monotonicity to 1.0 does
not.

**H4 (efficiency) — DESCOPED.** No GPU-hours-to-target-accuracy comparison and no GRPO run
(BUILDPLAN section 8 fallback 1). Stated as cut, not attempted.

**Honest caveats.** One task (synthetic pointer-chase), one model pair, 400 steps, greedy eval,
3 seeds; GSM8K wired but not run. B2 off-policy logit-KD stayed near floor (loss fell 0.57 -> 0.13
but accuracy ~0.1): mass-covering forward KL from a ~30%-accurate teacher does not sharpen into
correct greedy answers, whereas hard-label SFT does. Not tuned further (tau=1, lr 1e-5); reported
as observed.

## Phase log (no-run phases recorded here; numbered runs go below)

### 2026-07-05  Phase 4-5 — Full sweep COMPLETE (36/36)
- 12 conditions x 3 seeds on pointer-chase, all completed on Explorer V100-sxm2 (~50 GPU-hr,
  submitted in waves of 8 under the gpu QOS 8-submit/4-run cap). Results + verdicts above; figures
  in results/figures/. H1 confirmed (crossover + positional-KL mechanism), H2 accuracy confirmed
  (RKL>JSD>FKL) / diversity untested, H3 partially (saturates ~lambda 0.5-0.75), H4 descoped.

### 2026-07-02  Phase 4 — Cluster pipeline VALIDATED (smoke; NOT a result)
- Explorer (Northeastern) bring-up. Env: torch 2.4.1+cu121 (cu130 default was too new for the
  node driver, CUDA 12.3), transformers 4.57.6, datasets 5.0.0. Install debugging documented:
  login-node reaper (run install as a batch job), download.pytorch.org blocked by proxy (torch
  from PyPI), `module purge` clears the proxy (export after module load).
- Smoke run (configs/smoke_cluster.yaml, OPD-RKL, 20 steps, Qwen2.5-0.5B<-1.5B) COMPLETED in
  3:33 on a Tesla V100. The full real-model path works: load -> on-policy generate -> reverse-KL
  update (loss 0.582 -> 0.293) -> greedy horizon eval -> positional-KL probe -> metrics.json +
  figure. Two bugs the smoke caught and fixed: dtype= kwarg (transformers 5/4), and HFSampler
  passing example dicts to the tokenizer instead of prompt strings.
- These smoke metrics are a PLUMBING CHECK, not a finding: 20 steps cannot teach the task, so the
  accuracy (~0.1) is noise. No scientific number is claimed. The next step is a calibration run to
  size steps + GPU-hours, then the pre-registered sweep.

### 2026-07-02  Phase 4 — Task-viability diagnostic (zero-shot, greedy, n=30)
Before spending sweep compute, checked whether the teacher can actually do pointer-chase (else the
horizon study is floor-vs-floor). Critical finding: the Qwen2.5 Instruct models need the chat
template; the initial harness fed raw prompts.

| horizon k | teacher raw | teacher chat-templated | student base chat-templated |
|-----------|-------------|------------------------|-----------------------------|
| 2 | 0.033 | 0.367 | 0.033 |
| 3 | 0.000 | 0.400 | 0.067 |
| 4 | 0.067 | 0.167 | 0.100 |
| 5 | 0.067 | 0.200 | 0.067 |
| 6 | 0.067 | 0.300 | 0.033 |

- With the chat template the teacher goes from ~5% (floor) to ~30%, well above the ~6% base
  student. There is a real capability gap to distill: the task is viable. The earlier calibration
  (student stuck at ~8%) was crippled by the missing chat template, not a dead task.
- Harness fixes from this phase: apply_chat_template for Instruct models (+ add_generation_prompt);
  generation length 96 -> 256 (teacher CoT was truncated); torch 2.4.1+cu121 (cu130 default too new
  for the CUDA 12.3 node driver); request a 32GB GPU with batch 4 (full-vocab V=151936 divergence
  OOMs 16GB); batch the eval generation and the positional-KL probe (forwarding a whole horizon set
  or a 32-example probe OOMs even 32GB).
- GPU time: a 400-step OPD-RKL run is ~1h20m on a V100-sxm2 (chat template + 256-token generation).
  A 1000-step run projects to ~3.4h, near the 4-GPU-hour per-run checkpoint; sweep step count and
  the 36-run aggregate (~50-60+ GPU-hr) need a scoping decision before launch.

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
