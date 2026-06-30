# BUILD PLAN — kd-lab: On-Policy Distillation Branch
### Reproducing and stress-testing on-policy distillation to isolate the exposure-bias mechanism

> **What this file is.** A complete, self-contained runbook for a Claude Code agent loop. It defines the
> motivation, the exact technical design, the software architecture, an 8-phase execution loop with
> self-checkable acceptance gates, and the deliverables (code + results + a Medium/LinkedIn write-up).
> Upload it to a Claude Code session opened in the `kd-lab` repo. The agent reads the existing repo first,
> then extends it. Nothing here auto-pushes anything public without a human checkpoint.

> **Audience reality.** Parv built `kd-lab` himself (off-policy KD: Hinton logit KD, TinyBERT
> intermediate-layer + DistilBERT cosine, sequence-level Kim & Rush). He does not vibe-code research. Core
> math is implemented and understood by hand; third-party trainers are used only as correctness oracles.

---

## 0. Operating contract for the agent

Follow these rules for the entire run. They are not optional.

**Mode.** This is `/buildmode`. Plan, implement, test, evaluate, document. You are an engineer-scientist,
not a cheerleader. State assumptions. Flag guesses explicitly. When something is reproduction (not novel),
say so in code comments and in the write-up.

**Style (matches the lab charter).** No em dashes in prose. No filler words ("crucial", "leverage",
"robust", "seamless"). No "excited/passionate". Direct engineer prose. Cite prior art with arXiv ids.

**Honesty clause (hard rule).** Never fabricate or round-trip a result you did not actually produce. Every
number in a table or plot must come from a logged run with a config hash. If a run fails or a hypothesis
does not hold, report the negative result and analyze why. A clean negative result is a shippable result.

**Tool use.** Use everything available:
- **Filesystem + git.** Read the real `kd-lab` source before writing anything. Do not assume its structure.
  Extend its existing abstractions. Use feature branches and conventional commits. One coherent commit per
  completed phase.
- **GitHub.** Inspect history, open a draft PR `feat: on-policy distillation` at Phase 3, keep it updated.
- **Recall MCP (if connected).** Before reasoning from scratch, `search` Parv's Recall knowledge base for
  saved cards on: on-policy distillation, GKD, GRPO, reverse KL, sequence-level KD, Thinking Machines. Cite
  the saved card if you draw on it. His saved reading is signal.
- **Skills.** Use the relevant document skill to produce the Medium article as a clean Markdown deliverable.
  Use a plotting setup (matplotlib, no seaborn styling assumptions) for all figures.
- **Compute.** A100 via SLURM. Submit long runs as batch jobs, not interactive. Log everything to W&B (or
  the repo's existing logger) with the exact config.

**Loop protocol.** For each phase: (1) restate the goal and the acceptance criteria, (2) write tests first
where applicable, (3) implement, (4) run, (5) check the acceptance criteria yourself, (6) commit, (7) write
a 3-5 line entry to `EXPERIMENTS.md` (what ran, config hash, result, surprise). Do not advance to the next
phase until the current acceptance criteria pass. If you cannot pass a gate after two honest attempts, stop
and summarize the blocker for the human.

**Human checkpoints (stop and ask before doing these).**
- Before any training run estimated to exceed 4 GPU-hours.
- Before pushing any model, dataset, or repo public, or publishing the article anywhere.
- Before deleting or force-pushing anything.
- If the scope needs to shrink (see Section 8 fallbacks), propose the reduced scope and wait.

**Definition of done.** Section 13. Read it now so you build toward it.

---

## 1. Thesis and motivation (the reason this exists)

`kd-lab` currently implements only **off-policy** distillation: the student is trained on data drawn from
the *teacher's* distribution (teacher logits on a fixed corpus, or teacher-generated sequences). This has a
known structural flaw, **exposure bias** / train-inference mismatch: the student only ever sees states the
teacher visits, but at inference time it conditions on *its own* previously generated tokens, drifts into
states it was never trained on, and errors compound along the sequence. This is the same pathology that
**DAgger** (Ross, Gordon, Bagnell, 2011) was invented to fix in imitation learning: train the learner on
the *learner's own* state distribution with expert supervision, so it learns to recover from its own
mistakes.

**On-policy distillation** is the language-model instance of that fix. The student generates its own
rollouts; a frozen teacher scores every token of those rollouts with its full next-token distribution
(a dense signal); the student is updated to match the teacher on the states it actually visits. It sits
between SFT (off-policy, dense reward) and RL (on-policy, sparse scalar reward), and it is cheap and stable
because it is **supervised learning on self-generated data**, not policy gradient: there is no backprop
through sampling and no REINFORCE variance. This is exactly why Thinking Machines (Lu, Oct 2025) and the
Qwen3 team (2025) report frontier-level gains at a fraction of RL compute, building on GKD (Agarwal et al.,
2023). It is also the mechanism at the center of the current "get deployment learning back into the
weights cheaply" debate (Dwarkesh Patel, Jun 2026; Sutton; Karpathy).

**Why this is a real project and not a tutorial.** The method is established. The defensible work is the
*controlled isolation of why it works*, which the literature asserts but rarely shows cleanly at small
scale, plus an honest account of its failure mode:

1. **Mechanism isolation.** Measure the off-policy-vs-on-policy performance gap *as a function of generation
   horizon*. If exposure bias is the cause, the gap should widen with horizon, and off-policy
   teacher-student divergence should grow with token position while on-policy stays flatter. This is a
   direct test of the causal story, not a single leaderboard number.
2. **Design-space implementation.** Implement the full GKD design space inside `kd-lab`: the divergence
   family (forward KL, reverse KL, generalized JSD(beta)) crossed with the data source (off-policy teacher
   data, on-policy student rollouts, mixed by fraction lambda). Most public implementations expose one
   corner of this; a clean, tested, swappable implementation is genuine engineering value.
3. **Failure-mode characterization.** Reverse KL is mode-seeking (MiniLLM, Gu et al., 2023). Test the
   documented tradeoff: reverse-KL on-policy should improve greedy/pass@1 but may collapse diversity
   (pass@k for large k, token entropy, distinct-n). Report it whether or not it reproduces.

**Why Parv specifically.** Compounding error over a horizon is the identical concern in his `av-policy-lab`
closed-loop planning work (a diffusion policy's edge over a deterministic baseline in interaction-critical,
long-horizon scenarios). This branch lets him connect `kd-lab` (off-policy KD he already built) to the
imitation-learning exposure-bias literature and to closed-loop planning in one coherent narrative. That
cross-domain synthesis is the article's spine and the portfolio differentiator.

**This is simultaneously a reproduction of a famous result and an extension of an existing project.** It
reproduces GKD / on-policy distillation (famous) inside `kd-lab` (his repo), and adds the controlled
horizon analysis and failure-mode study (new at this scale, in one clean place).

---

## 2. Prior art (read before designing; cite in code and write-up)

- **Hinton, Vinyals, Dean, 2015** — Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
  Off-policy logit KD with temperature; the `kd-lab` baseline lineage.
- **Kim & Rush, 2016** — Sequence-Level Knowledge Distillation. arXiv:1606.07947. Off-policy: train student
  on teacher-generated sequences. A primary baseline to beat.
- **Ross, Gordon, Bagnell, 2011** — A Reduction of Imitation Learning and Structured Prediction to No-Regret
  Online Learning (DAgger). The conceptual parent of on-policy distillation.
- **Agarwal et al., 2023** — On-Policy Distillation of Language Models: Learning from Self-Generated
  Mistakes (GKD). arXiv:2306.13649; ICLR 2024. The divergence x data-source generalization implemented here.
- **Gu et al., 2023** — Knowledge Distillation of Large Language Models (MiniLLM). arXiv:2306.08543. Argues
  reverse KL for generative LMs; source of the diversity-tradeoff hypothesis.
- **Lu, Kevin & Thinking Machines Lab, Oct 2025** — On-Policy Distillation. DOI 10.64434/tml.20251026.
  Per-token reverse KL; the cheap-and-stable framing; Tinker cookbook reference implementation.
- **TRL `GKDTrainer` / `DistillationTrainer`** (HuggingFace) — used only as a correctness oracle for the
  hand-rolled loss and loop. Not the primary implementation.

If Recall has Parv's saved cards on any of these, read them first and note what framing he already absorbed.

---

## 3. Research questions and pre-registered hypotheses

Fix the primary metric and hypotheses *before* running, to avoid post-hoc cherry-picking.

- **RQ1 (mechanism).** Does the off-policy-vs-on-policy gap grow with generation horizon?
  - **H1.** On a horizon-controllable task, off-policy KD accuracy degrades faster with horizon than
    on-policy KD. **Primary metric:** accuracy vs horizon bucket; secondary: positional teacher-student KL
    vs token index over student rollouts.
- **RQ2 (divergence).** How does the divergence choice trade off correctness vs diversity?
  - **H2.** Reverse KL improves pass@1 over forward KL but reduces diversity (lower pass@k for k>=16, lower
    token entropy). JSD(beta=0.5) sits between.
- **RQ3 (data source).** How much on-policy data is needed?
  - **H3.** Performance improves monotonically (with diminishing returns) as the on-policy fraction lambda
    goes from 0 (off-policy) to 1 (fully on-policy); most of the gain arrives by lambda in [0.5, 1.0].
- **RQ4 (efficiency, stretch).** Does on-policy distillation reach a target accuracy at lower training
  compute than off-policy KD (and, if time permits, a small GRPO run)?
  - **H4.** On-policy distillation reaches the off-policy KD's best accuracy in fewer GPU-hours / training
    tokens.

---

## 4. Scope, contribution, and non-goals

**In scope.** Single-node, small models, one task family for the mechanism (synthetic, horizon-controllable)
plus one real task for external validity (GSM8K). Forward/reverse/JSD divergences. lambda sweep. Diversity
and efficiency analysis. Clean tested code in `kd-lab`. A reproducible repo and a write-up.

**The contribution, stated honestly (put this verbatim in the README and article).** "The on-policy
distillation method is not novel; it is GKD (Agarwal et al., 2023), popularized by Thinking Machines (2025).
This project's contribution is (1) a clean, tested implementation of the full divergence x data-source
design space inside an existing distillation library, (2) a controlled experiment isolating the
exposure-bias mechanism via horizon-stratified analysis, connecting LM distillation to imitation-learning,
and (3) an honest characterization of the reverse-KL diversity-collapse failure mode."

**Non-goals (do not attempt unless explicitly told).**
- Cross-tokenizer distillation (keep teacher and student in the same tokenizer family). Note as future work.
- Frontier-scale models, multi-node training, or beating any published SOTA number.
- Reproducing Subquadratic / SSA. Out of scope for this branch.
- Backprop through sampling or full RLHF. The point is the supervised, sample-free on-policy update.

---

## 5. Experimental design

### 5.1 Models (same tokenizer family to avoid cross-tokenizer issues)
- **Primary student:** `Qwen2.5-0.5B-Instruct` (full fine-tune; fits comfortably on one A100).
- **Primary teacher:** `Qwen2.5-1.5B-Instruct` (frozen). For GSM8K, optionally `Qwen2.5-Math-1.5B-Instruct`.
- **Optional larger teacher:** `Qwen2.5-3B-Instruct` for a bigger capability gap (more sampling/scoring cost;
  gate behind a checkpoint).
- Precision bf16. Optimizer AdamW, cosine schedule with warmup, gradient accumulation as needed. Seeds fixed.

### 5.2 Tasks
- **Synthetic, horizon-controllable (for RQ1 mechanism).** A k-step deterministic task where horizon = k is a
  dial. Pick one and justify it: multi-step modular arithmetic (chain of k operations), a k-hop key-value
  lookup, or a k-length copy-with-transform. Requirement: a single ground-truth answer, exact-match scorable,
  and difficulty controlled almost entirely by k. This is what makes the horizon plot clean.
- **GSM8K (for external validity, RQ1-RQ3).** Chain-of-thought math. Variable solution length gives a natural
  horizon stratification. Eval: exact match on the final numeric answer on the test split. Use the standard
  prompt format; document it.

### 5.3 Conditions
Baselines (off-policy, some already in `kd-lab`, reuse don't rewrite where possible):
- **B0 SFT** on gold (or teacher-generated) solutions.
- **B1 Sequence-level KD** (Kim & Rush): SFT on teacher greedy/beam generations.
- **B2 Off-policy logit KD** (Hinton-style forward KL on teacher data, temperature tau).

On-policy (new):
- **OPD-RKL** student rollouts + per-token reverse KL (primary; matches Thinking Machines).
- **OPD-FKL** student rollouts + forward KL.
- **OPD-JSD(beta)** student rollouts + generalized JSD, beta in {0.1, 0.5, 0.9}.
- **lambda sweep** mixed data source, lambda in {0.0, 0.25, 0.5, 0.75, 1.0} with the primary divergence.

### 5.4 Metrics
- **Primary:** task accuracy (exact match), greedy and sampled pass@1, with 95% CI over seeds.
- **Mechanism:** accuracy vs horizon bucket (synthetic: vs k; GSM8K: vs solution-length bucket); positional
  teacher-student KL averaged over student rollouts as a function of token index.
- **Diversity / failure:** pass@k for k in {1, 4, 16}, mean token entropy, distinct-1/distinct-2.
- **Efficiency:** accuracy vs cumulative GPU-hours and vs training tokens (OPD vs off-policy KD; optional GRPO).

### 5.5 Statistical protocol
- >= 3 seeds for every headline comparison; report mean +/- std or 95% CI (bootstrap).
- For the OPD-vs-off-policy gap, report a paired bootstrap CI on the per-example correctness difference.
- Pre-register the primary metric (Section 3). Do not switch it after seeing results.
- Log every run's full config and a content hash; figures reference run ids.

---

## 6. The loss math (implement by hand; this is the appendix you build from)

Let the student generate response tokens y = (y_1, ..., y_L) for prompt x (sampling temperature T_gen,
e.g. 1.0; top-p as configured). The generated sequence is treated as **fixed data**: no gradient flows
through sampling. Score it with both models by a forward pass over [x; y]:
- Student next-token distribution at position t: p_S(. | x, y_{<t}), differentiable (gradients flow here).
- Teacher next-token distribution at position t: p_T(. | x, y_{<t}), frozen (no grad).

Apply an optional distillation temperature tau to both logits before softmax (default tau = 1.0 for OPD;
classic Hinton KD uses tau > 1 with a tau^2 loss scaling). Compute a per-token divergence over the vocab and
average over response positions only (mask the prompt and pad).

- **Forward KL** (teacher || student), mass-covering:
  D_FKL(t) = sum_v p_T(v) * [log p_T(v) - log p_S(v)]
- **Reverse KL** (student || teacher), mode-seeking (primary):
  D_RKL(t) = sum_v p_S(v) * [log p_S(v) - log p_T(v)]
- **Generalized JSD(beta)**, beta in [0,1], with m = beta * p_T + (1 - beta) * p_S:
  D_JSD(t) = beta * KL(p_T || m) + (1 - beta) * KL(p_S || m)
  Note the correct behavior: JSD(beta) is 0 at beta in {0,1} and is maximally symmetric near beta = 0.5; it
  is a bounded, symmetric-family interpolation, **not** literally equal to forward/reverse KL at the
  endpoints. Use it as a stability-vs-sharpness knob and cite Agarwal et al. (2023) for the generalized
  treatment. Do not claim the endpoints equal the raw KLs.

Per-sequence loss = mean over valid response positions of D(t). Numerical stability: compute in log-space
with `log_softmax`; never exponentiate raw logits; mask with a boolean response mask; clamp where needed.

**Implementation sanity checks (turn into unit tests):**
- D_RKL(p, p) == 0 and D_FKL(p, p) == 0 within tolerance.
- D_JSD(beta=0) == 0 and D_JSD(beta=1) == 0 within tolerance; D_JSD symmetric in (p_T, p_S) at beta=0.5.
- Gradient of D_RKL w.r.t. student logits has the expected sign on a 2-token toy vocab.
- Masking: positions in the prompt and pad contribute exactly zero to the loss and the gradient.
- Loss matches TRL `GKDTrainer` on a tiny fixed config within tolerance (correctness oracle).

---

## 7. Software architecture (extend `kd-lab`; do not fork a parallel system)

**Phase 0 first.** Read the existing repo. Identify its base classes (likely a `Distiller`/`Trainer`, loss
modules, a config system, an eval harness). Map exactly where on-policy plugs in. Reuse the existing
off-policy baselines and config/logging. Write `DESIGN.md` recording the integration points before any code.

Target abstractions (adapt names to the repo's conventions):

```
kd_lab/
  divergences/
    base.py           # Divergence(ABC): forward(student_logits, teacher_logits, mask, tau) -> loss
    forward_kl.py     # ForwardKL
    reverse_kl.py     # ReverseKL
    generalized_jsd.py# GeneralizedJSD(beta)
  data_sources/
    base.py           # RolloutSource(ABC): batch() -> sequences (+ teacher cache hooks)
    teacher_source.py # TeacherRolloutSource (off-policy; precomputed teacher generations)
    student_source.py # StudentRolloutSource (on-policy; generate from current student)
    mixed_source.py   # MixedRolloutSource(lambda)  # mixes the two by fraction lambda
  sampling/
    base.py           # Sampler(ABC)
    hf_sampler.py     # HF generate
    vllm_sampler.py   # vLLM-backed fast sampling (preferred for on-policy throughput)
  distillers/
    base.py           # existing Distiller base
    off_policy.py     # existing / reused baselines (SFT, seq-KD, logit KD)
    on_policy.py      # OnPolicyDistiller: generate -> teacher-score -> divergence -> update
  evaluation/
    gsm8k.py          # exact-match accuracy
    horizon.py        # HorizonStratifiedEvaluator (synthetic k; GSM8K length buckets)
    diversity.py      # pass@k, distinct-n, token entropy
    positional_kl.py  # teacher-student KL vs token index probe
  configs/            # dataclass or repo-native configs (+ YAML); one file per condition; seeded
  experiments/        # SLURM submit scripts + a runner that sweeps conditions/seeds
  tests/              # pytest; the Section 6 sanity checks + a tiny end-to-end smoke test
```

**Engineering practices to enforce.**
- OOP: small single-responsibility classes behind ABCs; dependency injection of divergence, sampler, and
  data source into the distiller; no hard-coded model names.
- Type hints everywhere; `ruff` lint; `mypy` type-check; Google-style docstrings on public APIs.
- Determinism: seed everything (Python, NumPy, torch, CUDA); log seeds and config hash with each run.
- Config-driven runs (no magic constants in code); every figure traceable to a config.
- `pytest` with a fast smoke test on a 2-layer toy model so CI runs without a GPU.
- GitHub Actions CI: tests + lint + type-check on push. Pre-commit hooks.
- Logging via the repo's logger or W&B; persist student generations as artifacts for later analysis.
- The on-policy loop must reuse the teacher forward pass efficiently (no redundant scoring), generate with
  vLLM where possible, and support gradient accumulation for small-GPU configs.

---

## 8. Execution loop (8 phases, each with a self-check gate)

Run sequentially. Do not skip gates. Write an `EXPERIMENTS.md` entry per phase.

**Phase 0 — Orient.** Read `kd-lab`. Search Recall for Parv's saved OPD/GKD/KD cards. Read GKD (2306.13649)
and the Thinking Machines blog. Produce `DESIGN.md`: current abstractions, integration points, the exact
loss math, the condition matrix, the metric definitions.
*Gate:* `DESIGN.md` exists, names real integration points in the repo, no code yet.

**Phase 1 — Divergences + tests.** Implement the divergence family with all Section 6 unit tests.
*Gate:* `pytest` green; all numerical and gradient checks pass.

**Phase 2 — Off-policy baseline parity.** Wire B0/B1/B2 (reuse existing code) into the new eval harness.
Train the 0.5B student to a sane GSM8K baseline.
*Gate:* baselines run end-to-end on the A100; accuracy in a plausible range; metrics logged with configs.

**Phase 3 — On-policy distiller.** Implement generate -> teacher-score -> reverse-KL update with the
sampler and data-source abstractions. Validate the loss against TRL `GKDTrainer` on a tiny fixed config.
Open the draft PR.
*Gate:* OPD loop trains, loss decreases, matches the TRL oracle within tolerance on the smoke config.

**Phase 4 — Core experiment (RQ1).** Run baselines vs OPD-RKL on the synthetic horizon task and GSM8K,
>= 3 seeds. Produce the headline horizon plot and the positional-KL probe.
*Gate:* results table with CIs; horizon plot exists; H1 confirmed or refuted in writing.

**Phase 5 — Ablations (RQ2, RQ3).** Divergence sweep (FKL/RKL/JSD beta), lambda sweep, diversity metrics
(pass@k, entropy, distinct-n).
*Gate:* ablation tables/plots; the reverse-KL diversity-collapse hypothesis reported either way.

**Phase 6 — Efficiency (RQ4, optional).** Accuracy vs GPU-hours / tokens for OPD vs off-policy KD; a small
GRPO run only if time and compute allow (behind a checkpoint).
*Gate:* efficiency plot, or an explicit note that it was descoped and why.

**Phase 7 — Write-up and polish.** Synthesize results into the README results section; draft the Medium
article and the LinkedIn post (Section 9); finalize all figures; green CI; full docstrings; reproduction
commands. Final commit; update the PR.
*Gate:* Section 13 definition of done is fully satisfied.

**Fallback scopes (propose at a checkpoint if time/compute is tight, in this order):**
1. Drop the optional larger teacher and the GRPO efficiency run.
2. Reduce seeds from >=3 to 2 and reduce the GSM8K eval to a fixed 500-example subset (documented).
3. Keep only the synthetic horizon task for RQ1 and use GSM8K only for a single headline OPD-vs-SFT number.
4. Minimum shippable core: B0/B1 baselines + OPD-RKL + the horizon plot + the positional-KL probe + the
   write-up. This alone is a complete, honest project.

---

## 9. Deliverables

1. **Code** on a `feat/on-policy-distillation` branch, merged via the PR after review. Tested, typed, CI green.
2. **README** updated with: the honest contribution statement (Section 4), the results table, the headline
   horizon figure, and one-command reproduction instructions.
3. **`EXPERIMENTS.md`** run log; **`results/`** with figures (PNG + the data behind them) and tables (CSV).
4. **Medium article** as a clean Markdown file in `writeup/`. Spec below.
5. **LinkedIn post** (short) in `writeup/linkedin.md`.
6. **Optional, behind a checkpoint:** push the distilled student + a dataset card to the HF Hub.

### 9.1 Medium article spec (honest, technical, cross-domain)
- **Title direction (pick one, no hype):** "On-policy distillation, and what it borrows from DAgger" /
  "Why training a student on its own mistakes fixes exposure bias: a controlled study".
- **Structure:** (1) the exposure-bias problem in off-policy KD, with the DAgger parallel; (2) what on-policy
  distillation is, with the exact reverse-KL math and the key point that it is supervised on self-generated
  data, not policy gradient; (3) the controlled experiment and the horizon result; (4) the divergence and
  diversity tradeoff (the failure mode); (5) the connection to closed-loop planning / `av-policy-lab`
  (compounding error over a horizon is the same animal); (6) honest limitations and what was reproduction
  vs new. 1500-2500 words. Every figure from the repo. Link the repo and cite GKD, MiniLLM, Thinking
  Machines, DAgger with arXiv ids.
- **Claims discipline:** do not say "I invented". Say "I reproduced GKD inside my distillation library and
  ran a controlled study isolating the mechanism." Report negative results plainly.

### 9.2 LinkedIn post spec
4-8 sentences. The one-line hook (training a student on its own mistakes vs the teacher's), the single most
interesting plot, one honest caveat, the repo + article links. No emoji-spam. No "thrilled/excited".

---

## 10. Risks and mitigations
- **On-policy sampling is slow** -> use vLLM for generation; cache teacher scores; small student (0.5B);
  gradient accumulation; subset eval during development.
- **Training instability at high lambda or with reverse KL** -> warm up from an SFT checkpoint; clip; tune
  T_gen and learning rate; this instability is itself a documented finding to report, not just a bug.
- **Horizon signal is noisy on GSM8K** -> that is why the synthetic horizon-controllable task exists; it
  carries RQ1 cleanly while GSM8K provides external validity.
- **Scope creep** -> the fallback ladder in Section 8; the minimum shippable core is defined.
- **Overclaiming** -> the honesty clause and the verbatim contribution statement.

---

## 11. Compute and time budget (honest estimate; measure and update)
Single A100 (40 or 80 GB). 0.5B full fine-tune with on-policy generation on GSM8K-scale: roughly a few
GPU-hours per run. ~8-12 runs across baselines, divergence/lambda ablations, and seeds: a weekend of
wall-clock with overnight batch jobs. Report exact GPU-hours per run in `EXPERIMENTS.md`. If a single run
trends past 4 GPU-hours, stop and check in.

---

## 12. Quick start for the human (after the agent finishes Phase 0)
```bash
# branch
git checkout -b feat/on-policy-distillation
# env
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # adapt to the repo's setup
pytest -q                        # Section 6 sanity checks must pass before training
# a single condition, smoke size, to confirm the loop
python -m kd_lab.experiments.run --config configs/opd_rkl_smoke.yaml
```

---

## 13. Definition of done (the agent checks every box before declaring completion)
- [ ] `DESIGN.md` written from the real repo; integration points named.
- [ ] Divergence family implemented; all Section 6 unit tests pass; CI green (tests + ruff + mypy).
- [ ] Off-policy baselines reused and runnable through the new eval harness.
- [ ] On-policy distiller implemented; loss validated against TRL `GKDTrainer` on a smoke config.
- [ ] RQ1 horizon experiment run with >= 2-3 seeds; headline horizon figure + positional-KL probe produced.
- [ ] At least the divergence and lambda ablations done (or the fallback core, with the cut documented).
- [ ] Results table with confidence intervals; every number traceable to a logged config.
- [ ] README updated with the honest contribution statement, results, and reproduction commands.
- [ ] Medium article draft and LinkedIn post draft in `writeup/`, figures from the repo, prior art cited.
- [ ] `EXPERIMENTS.md` complete; PR updated; nothing public pushed without a human checkpoint.
- [ ] Every hypothesis (H1-H4) reported as confirmed, refuted, or descoped, with reasoning. Negative results
      kept.

---

*Final reminder to the agent: the value of this project is honesty and mechanism, not a leaderboard number.
A clean reproduction plus a controlled isolation of why on-policy distillation works, plus an honest failure
mode, is the deliverable. If at any point you are tempted to round a number up or claim novelty you did not
earn, stop and write down what actually happened instead.*
