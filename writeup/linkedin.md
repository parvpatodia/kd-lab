# LinkedIn post (template)

Keep it 4-8 sentences. One hook, one result, one honest caveat, the links. No emoji spam, no
"thrilled/excited". Fill the brackets after the run.

---

Off-policy distillation trains a small model on the teacher's outputs. On-policy distillation
trains it on its own outputs, graded token by token by the teacher. The second one fixes a
specific failure: when a model conditions on its own earlier tokens at inference, errors compound
over the sequence.

I added on-policy distillation to my distillation library and ran a controlled test of that
mechanism: I measured the off-policy-versus-on-policy gap as a function of how many reasoning
steps the task requires. [Result: the gap grew from X at k=2 to Y at k=6 / the gap did not grow,
which suggests Z.] I also reproduced the reverse-KL diversity tradeoff: [one line].

This is a reproduction of GKD (Agarwal et al., 2023), not a new method. The point was to
understand why it works and to connect it to the compounding-error problem I study in closed-loop
planning. Write-up and code below.

[article link] · [github link]
