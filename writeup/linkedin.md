# LinkedIn post

I ran a controlled study on on-policy distillation: instead of training a small student on a
teacher's fixed outputs, the student generates its own rollouts and a frozen teacher grades every
token, so the student learns on the states it actually visits at inference. It is the language-model
version of DAgger's fix for exposure bias.

On a horizon-controllable task (train on short chains, test on longer ones), the result was a clean
crossover. Off-policy SFT was near-perfect in-distribution and then collapsed on longer horizons
(0.99 to 0.04); on-policy reverse-KL distillation was weaker in-distribution but held up on
extrapolation (0.21 at the longest horizon, a ~5x gap). A per-token teacher-student KL probe shows
why: the off-policy student drifts away from the teacher on its own rollouts, while the on-policy
student stays aligned.

Honest caveat: this is one synthetic task and greedy evaluation. I did not yet test the known
reverse-KL diversity-collapse tradeoff (pass@k, entropy), and the efficiency comparison is future
work. The method is not mine, it is GKD (Agarwal et al., 2023); the contribution is the clean
divergence-by-data-source implementation and the controlled mechanism study.

Same compounding-error-over-horizon problem I see in closed-loop planning, which is why I built it.
Code, 36-run results with config hashes, and figures: https://github.com/parvpatodia/kd-lab
