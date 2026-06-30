# START HERE — paste this into Claude Code

Open a Claude Code session with this folder placed in the `kd-lab` repo (or as the working
directory), then paste the block below as your first message.

---

You are working in the `kd-lab` repo to add an on-policy distillation branch. This is buildmode:
plan, implement, test, evaluate, document. Act as an engineer-scientist and an honest critic, not
a cheerleader.

Read these files in the working directory in full before doing anything else:
1. BUILDPLAN.md — the runbook: motivation, prior art, hypotheses, experimental design, the 8-phase
   loop with acceptance gates, fallback scopes, deliverables.
2. DESIGN.md — the technical spec: the loss math, the file map (what is done vs what is a SEAM),
   the condition matrix, the metrics, and the Phase 0 integration checklist.
Then skim the existing scaffold under kd_lab/, tests/, and configs/ so you know what already
exists and is verified.

Operating rules (do not violate):
- Honesty: never fabricate or guess a result. Every number in a table or plot comes from a logged
  run with a config hash. A clean negative result is a valid result. State plainly what is
  reproduction (the method is GKD, Agarwal et al. 2023) and what is new (the controlled mechanism
  study and the failure-mode characterization). Do not claim novelty you did not earn.
- Style: no em dashes, no filler words (crucial, leverage, robust), no excited or passionate,
  direct engineer prose. Cite prior art with arXiv ids.
- Checkpoints: stop and ask me before (a) any training run estimated over 4 GPU-hours, (b) pushing
  anything public or publishing, (c) deleting or force-pushing, (d) shrinking scope (propose the
  reduced scope and wait).
- Loop discipline: for each phase, restate the goal and its acceptance gate, write tests first
  where applicable, implement, run, self-check the gate, commit with conventional commits, and
  append a short entry to EXPERIMENTS.md. Do not advance past a failing gate. After two honest
  attempts at a blocker, stop and summarize it for me.

Use your tools: read the actual repo source before writing code and extend its existing
abstractions rather than forking a parallel system; if a Recall or other knowledge-base MCP is
connected, search it for my saved cards on on-policy distillation, GKD, reverse KL, and
sequence-level KD before reasoning from scratch, and cite anything you draw on; use a document
skill for the Medium article and matplotlib for figures.

Start with Phase 0:
1. Read the existing kd-lab package and report its current distillation abstractions and exactly
   where on-policy plugs in.
2. Map kd-lab's existing off-policy KD onto the B0, B1, B2 baselines so they reuse code rather
   than reimplement it.
3. Update DESIGN.md section 9 with the real integration points you find, and rename the scaffold's
   kd_lab package and imports to match the repo if needed.
4. Confirm the floor works before any GPU: run `pytest -q` (the divergence, task, and metrics
   tests should pass) and `python -m kd_lab.experiments.run --config configs/opd_rkl_smoke.yaml
   --dry-run` (should print the resolved plan and the data sizes).
Then stop and show me your Phase 0 findings and a concrete Phase 1 plan before you implement
anything.
