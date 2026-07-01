# DESIGN — kd-lab on-policy distillation branch

This is the technical reference the build plan's Phase 0 produces. It assumes the scaffold in
this folder is dropped into the `kd-lab` package (rename `kd_lab` to the repo's package name
and fix imports). Files marked **done** are complete and verified; files marked **SEAM** need
wiring to kd-lab's existing model/data utilities.

## 1. Question and contribution
`kd-lab` does only off-policy distillation, which trains the student on the teacher's
distribution and therefore suffers exposure bias: at inference the student conditions on its
own tokens, drifts into unseen states, and errors compound over the sequence. On-policy
distillation (GKD; Agarwal et al., 2023, arXiv:2306.13649) is the DAgger-style fix: the student
generates its own rollouts and a frozen teacher grades every token with its full distribution.

**Contribution, stated honestly (goes verbatim in README and article):** the method is not
novel (GKD 2023; Thinking Machines 2025). This branch contributes (1) a clean, tested
implementation of the full divergence x data-source design space inside an existing
distillation library, (2) a controlled experiment isolating the exposure-bias mechanism via a
horizon-stratified analysis that connects LM distillation to imitation learning, and (3) an
honest characterization of the reverse-KL diversity-collapse failure mode.

## 2. Loss math (implemented in `distillation/divergences.py`, **done**)
Student generates response tokens for prompt x. The rollout is fixed data (no gradient through
sampling). Score it with both models by a forward pass over [x; y]; gradients flow only through
the student; the teacher is detached.

- Forward KL (teacher||student), mass-covering: `sum_v p_T(v) (log p_T(v) - log p_S(v))`.
- Reverse KL (student||teacher), mode-seeking, primary: `sum_v p_S(v) (log p_S(v) - log p_T(v))`.
- Generalized JSD(beta): `beta*KL(p_T||m) + (1-beta)*KL(p_S||m)`, `m = beta*p_T + (1-beta)*p_S`.
  JSD(0)=JSD(1)=0, symmetric at beta=0.5, bounded by ln 2. It is **not** equal to the raw KLs at
  the endpoints; treat beta as a stability-vs-sharpness knob (Agarwal et al., 2023).

Verified properties (numpy reference + brute force, see `tests/test_scaffold.py`): self-KL = 0,
KL >= 0, JSD endpoints/symmetry/bound, reverse != forward, temperature softens, masking selects
only response tokens, gradient reaches the student and not the teacher.

**Critical detail:** the optimizer step is supervised learning on self-generated data. There is
no policy gradient and no REINFORCE term. That is why it is stable and cheap relative to RL.

## 3. Data-source axis (implemented in `distillation/on_policy.py`, **done logic / SEAM I/O**)
`MixedRolloutSource(on, off, lambda)` interpolates: lambda=0 is off-policy KD, lambda=1 is fully
on-policy. `shift_for_next_token` aligns logits[:, :-1] with response_mask[:, 1:] (verified). The
`Sampler` (HF `generate` or vLLM) and the model `.logits` calls are SEAMs.

## 4. Tasks
- **Pointer-chase** (`tasks/pointer_chase.py`, **done**): the horizon-controllable mechanism task.
  A random permutation table in-context, a start node, k hops. Horizon = k is one dial; reasoning
  length scales with k; a wrong hop propagates to all downstream hops (the exposure-bias setting);
  exact-match scorable with per-hop diagnostics. This carries RQ1 because the off-policy student
  under-covers the recovery states the on-policy student practices. Verified: determinism, the
  gold chain follows the table, the verifier round-trips, the horizon dial, error propagation.
- **GSM8K** (SEAM): real chain-of-thought math for external validity. Stratify by solution length
  for the horizon analysis. Use kd-lab's existing dataset utilities to load it.

## 5. Conditions (the experiment matrix)
Baselines (off-policy; reuse kd-lab code where it exists):
- B0 SFT on gold/teacher solutions; B1 sequence-level KD (Kim & Rush, 1606.07947); B2 off-policy
  forward-KL logit KD (Hinton, 1503.02531).
On-policy (new): OPD-RKL (primary), OPD-FKL, OPD-JSD(beta in {0.1, 0.5, 0.9}), and a lambda sweep
in {0.0, 0.25, 0.5, 0.75, 1.0} with the primary divergence.

## 6. Metrics (implemented in `evaluation/`, **done**; model-scored parts SEAM)
- Primary: exact-match accuracy, greedy and sampled pass@1, 95% CI over >= 3 seeds.
- Mechanism: accuracy vs horizon bucket (`metrics.horizon_stratified_accuracy`); positional
  teacher-student KL vs token index (`evaluation/positional_kl.py`).
- Diversity / failure (H2): pass@k for k in {1,4,16} (`metrics.pass_at_k`, unbiased, verified
  against brute force), distinct-n, empirical token entropy.
- Efficiency (H4, optional): accuracy vs GPU-hours / training tokens; optional small GRPO baseline.
- Statistics: `metrics.bootstrap_ci` and `metrics.paired_bootstrap_diff` (paired on the OPD-vs-
  baseline per-example correctness). Pre-register the primary metric; do not switch it post hoc.

## 7. File map
```
kd_lab/distillation/divergences.py   done   ForwardKL, ReverseKL, GeneralizedJSD, build_divergence
                                            (matches TRL GKD loss exactly, see tests/test_trl_oracle.py)
kd_lab/distillation/on_policy.py     done   Rollouts, Sampler(ABC), RolloutSource(+Mixed),
                                            shift_for_next_token, OnPolicyDistiller (CPU-smoke verified)
kd_lab/distillation/supervised.py    done   cross_entropy_next_token + SupervisedDistiller (B0/B1)
kd_lab/distillation/sampling.py      done   build_response_mask (exhaustively tested) + HFSampler
kd_lab/tasks/pointer_chase.py        done   generator + verifier + dataset builders
kd_lab/evaluation/metrics.py         done   pass@k, distinct-n, entropy, bootstrap, paired, horizon
kd_lab/evaluation/positional_kl.py   done   mechanism probe (wired into run.evaluate)
kd_lab/experiments/run.py            impl   full loop+eval; run_condition CPU-validated by injection.
                                            Remaining SEAMs: load_models real-HF path (cluster-only,
                                            untested locally: no CUDA + HF downloads blocked here) and a
                                            vLLM sampler for throughput. GSM8K loader not yet added.
pyproject.toml                       done   installable package + ruff/mypy/pytest config
configs/opd_rkl_smoke.yaml           done   tiny end-to-end smoke config
tests/test_scaffold.py               done   divergences, task, metrics (delivered suite)
tests/test_phase1.py                 done   gradient sign, numpy reference, tau^2, masking, CPU step
tests/test_supervised.py             done   B0/B1 CE objective + distiller
tests/test_sampling.py               done   response-mask cases + HFSampler wiring
tests/test_runner.py                 done   full sft/on_policy orchestration via injection (no HF/GPU)
tests/test_trl_oracle.py             done   divergence parity vs TRL GKD (skips if trl absent)
```

## 8. Models and compute
Student Qwen2.5-0.5B-Instruct (full fine-tune, bf16), teacher Qwen2.5-1.5B-Instruct (same
tokenizer family; no cross-tokenizer handling). Single A100. Use vLLM for student sampling to
keep on-policy throughput acceptable. Rough budget: a few GPU-hours per run; a weekend of
overnight jobs for baselines + ablations + seeds. Measure and record actual GPU-hours per run.
Stop and check with the human before any run estimated over 4 GPU-hours.

## 9. Phase 0 integration findings (real repo state, verified 2026-06-30)

**Repo state.** This folder *is* the kd-lab on-policy branch; the package is already `kd_lab`. No
separate legacy kd-lab with off-policy KD (Hinton logit KD, TinyBERT, DistilBERT cosine, Kim &
Rush seq-level) exists on disk, and the project notes record kd-lab as greenfield. So the earlier
"rename the package" and "reuse kd-lab's existing off-policy KD" steps do not apply as written. No
rename was performed (none needed). The baselines reuse the scaffold's *own* abstractions, below.

**Floor verified before any GPU.** `pytest -q` = 26 passed (torch 2.10.0 present, so the divergence
tests ran, not just the pure-Python ones). `python -m kd_lab.experiments.run --config
configs/opd_rkl_smoke.yaml --dry-run` prints the resolved plan and data sizes (n_train=512, eval
horizons 1..6 at 100 each, divergence_ok).

**Saved cards.** No Recall/knowledge-base cards on OPD/GKD/reverse KL/seq-level KD: the memory
graph is empty and the Feynman-Loop map holds only unrelated concepts. The design is from the
cited papers and this scaffold, not from saved reading. Stated plainly so the writeup does not
imply otherwise.

**Git blocker (open).** The folder is not its own repo. `.git` is at `/Users/parvpatodia/.git`
(repo rooted at home), remote `origin` = `parvpatodia/Memories.ai-Hackathon-`, and the scaffold
files are untracked. The feature-branch / commit / PR workflow is blocked until this is resolved
(recommended fix: `git init` a standalone repo in this folder, as `push_to_github.sh` already
assumes). No git changes were made. This needs a human decision.

**Baseline mapping (B0/B1/B2 onto the scaffold's abstractions; there is no prior code to reuse).**
- **B2** off-policy logit KD (Hinton, 1503.02531) = `OnPolicyDistiller` + `ForwardKL(temperature=tau)`
  + `TeacherRolloutSource`. The distiller is data-source-agnostic (off-policy = lambda 0 or a
  teacher source), so B2 is config-only, zero new loss code. Cleanest reuse.
- **B0** SFT = cross-entropy on gold response tokens. The `Divergence` interface needs a teacher
  distribution, so B0 needs a small hard-label objective (CE on target token ids, masked to the
  response); no teacher forward pass.
- **B1** sequence-level KD (Kim & Rush, 1606.07947) = the same hard-label CE objective, with data =
  teacher greedy/beam generations (the off-policy dataset builder).
- **OPD-RKL / OPD-FKL / OPD-JSD** = `OnPolicyDistiller` + the matching `Divergence` +
  `StudentRolloutSource` (lambda=1) or `MixedRolloutSource` (lambda<1). Already wired.

**Concrete SEAMs to implement (Phases 2-3), in `kd_lab/experiments/run.py` unless noted.**
1. `load_models`: Qwen2.5-0.5B-Instruct (student, trainable, bf16) + Qwen2.5-1.5B-Instruct (teacher,
   frozen) + shared tokenizer via transformers. Assert identical vocab size before scoring (the
   divergences assume an aligned vocab axis V).
2. `build_sampler`: a concrete `Sampler` returning `Rollouts` with a correct `response_mask` (1 on
   generated tokens only). HF `generate` for the smoke path (simple, correct, slow); vLLM with
   periodic weight-sync from the training copy is a Phase 4 throughput optimization, behind the
   4-GPU-hr checkpoint. The mask is what makes the loss and the probes correct.
3. `build_offpolicy_dataset`: precompute teacher greedy generations (B1) and gather gold targets
   (B0) as a dataset exposing `.batch(prompts) -> Rollouts` for `TeacherRolloutSource`.
4. Hard-label CE objective for B0/B1 (new, small): CE on target token ids, masked to the response,
   reusing `shift_for_next_token`. Sibling of `Divergence` or a thin SFT step; decide in Phase 2.
5. `build_optimizer`: AdamW + cosine schedule with warmup from `cfg['optim']`.
6. Step loop + evaluation in `train_and_eval`: batch prompts, run `distiller.step` to `max_steps`,
   log loss + config hash, then horizon-stratified accuracy, pass@k / entropy / distinct-n, and the
   positional-KL probe on `eval_sets`.
7. GSM8K loader + solution-length bucketing for the horizon analysis (external validity).

**Phase 3 oracle note.** Validate the hand-rolled loss against TRL `GKDTrainer` on forward/reverse
KL, where conventions are unambiguous. TRL's generalized-JSD parameterization must be matched
explicitly before comparing JSD; do not assume the endpoints equal the raw KLs (Section 2).

## 10. Hypotheses (pre-registered; see BUILDPLAN Section 3)
H1 off-policy degrades faster with horizon than on-policy. H2 reverse KL improves pass@1 but
lowers diversity vs forward KL. H3 gains rise with lambda, most arriving by 0.5-1.0. H4 on-policy
reaches the off-policy best accuracy in fewer GPU-hours. Report each as confirmed, refuted, or
descoped, with reasoning. Negative results are kept.
