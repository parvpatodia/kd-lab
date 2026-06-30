# Medium article outline

Target 1500-2500 words. Technical, honest, no hype. Every figure comes from the repo. Cite GKD,
MiniLLM, Kim & Rush, Hinton, DAgger, and the Thinking Machines post with arXiv ids / DOI. Write
in plain engineer prose: no em dashes, no "crucial/leverage/robust", no "excited/passionate".

## Working titles (pick one)
- On-policy distillation, and what it borrows from DAgger
- Why training a student on its own mistakes fixes exposure bias: a controlled study

## Contribution statement (paste verbatim near the top and in the README)
> The on-policy distillation method here is not novel. It is GKD (Agarwal et al., 2023),
> popularized by Thinking Machines (2025). This project contributes a clean, tested
> implementation of the full divergence x data-source design space inside my distillation
> library, a controlled experiment that isolates the exposure-bias mechanism by measuring how the
> off-policy-versus-on-policy gap grows with generation horizon, and an honest characterization of
> the reverse-KL diversity-collapse failure mode.

## Sections
1. **The problem with off-policy KD.** Teacher-forcing trains the student on the teacher's
   states; at inference it sees its own. Define exposure bias and compounding error. Draw the
   DAgger parallel (Ross et al., 2011): train on the learner's own state distribution.
2. **What on-policy distillation is.** Student rolls out, frozen teacher scores every token with
   its full distribution, student matches it. Give the reverse-KL formula. State the key point:
   this is supervised learning on self-generated data, not policy gradient, which is why it is
   stable and cheap. Place it between SFT and RL.
3. **The controlled experiment.** Describe pointer-chase and why horizon = k isolates the
   mechanism (a wrong hop propagates). [FIGURE 1: accuracy vs horizon, off-policy vs on-policy,
   with CIs.] [FIGURE 2: positional teacher-student KL vs token index.] Report whether the gap
   widens with horizon. Add the GSM8K result for external validity.
4. **The divergence and the failure mode.** Reverse vs forward KL vs JSD(beta). [FIGURE 3:
   accuracy and pass@k / entropy across divergences.] Report the reverse-KL diversity tradeoff
   honestly, present or absent.
5. **The connection to closed-loop planning.** Compounding error over a horizon is the same
   problem in `av-policy-lab` (a diffusion policy's edge over a deterministic baseline in long,
   interaction-critical scenarios). One paragraph tying distillation, imitation learning, and
   planning together.
6. **Limitations and what was reproduction.** Small models, one synthetic task plus GSM8K, no
   cross-tokenizer, no frontier claim. Separate clearly what was reproduced from what was new
   (the controlled isolation and the failure-mode study).

## Claims discipline
- Do not write "I invented". Write "I reproduced GKD inside my library and ran a controlled study
  of the mechanism."
- Every quantitative claim links to a figure or table backed by a logged run.
- Report negative results in the body, not a footnote.

## Repo link and reproduction
End with the GitHub link and the one-command smoke + full-run instructions from the README.
