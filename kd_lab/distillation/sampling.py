"""Concrete samplers that turn prompts into ``Rollouts`` for the on-policy loop.

The subtle, correctness-critical part is the ``response_mask``: the loss and the positional-KL
probe score only generated tokens, so the mask must be 1 on real generated tokens and 0 on the
prompt and on any post-EOS padding. That logic is isolated in ``build_response_mask`` (a pure
function, unit-tested) and reused by ``HFSampler``.

``HFSampler`` uses HuggingFace ``generate`` and is correct but slow; it is the smoke path. A
vLLM-backed sampler with periodic weight-sync from the training copy is the throughput path for
the A100 (DESIGN.md section 9) and is intentionally not implemented here.
"""

from __future__ import annotations

import torch

from .on_policy import Rollouts, Sampler


def build_response_mask(
    full_ids: torch.Tensor, prompt_len: int, eos_token_id: int | None
) -> torch.Tensor:
    """Response mask for left-padded batched generation.

    Assumes all prompts are left-padded to ``prompt_len`` (so generation begins at column
    ``prompt_len`` for every row). A generated position is a response token up to and including
    the first EOS; positions after the first EOS (trailing pad) are 0. This is correct even when
    ``pad_token_id == eos_token_id``. With ``eos_token_id is None`` every generated position counts.

    Args:
        full_ids: ``[B, T]`` prompt+generation token ids.
        prompt_len: number of left-padded prompt columns.
        eos_token_id: end-of-sequence id, or None.

    Returns:
        ``[B, T]`` long tensor, 1 on response tokens.
    """
    b, t = full_ids.shape
    mask = torch.zeros((b, t), dtype=torch.long, device=full_ids.device)
    if prompt_len >= t:
        return mask
    gen = full_ids[:, prompt_len:]
    if eos_token_id is None:
        mask[:, prompt_len:] = 1
        return mask
    is_eos = gen.eq(eos_token_id).long()
    eos_before = is_eos.cumsum(dim=1) - is_eos  # 0 up to and including the first EOS, then >= 1
    mask[:, prompt_len:] = (eos_before == 0).long()
    return mask


class HFSampler(Sampler):
    """Batched sampling via HuggingFace ``generate`` with left padding.

    Args:
        tokenizer: a HuggingFace tokenizer (padding side is forced to left).
        max_new_tokens: generation cap.
        temperature: sampling temperature; <= 0 selects greedy decoding.
        top_p: nucleus sampling parameter (used only when sampling).
        device: device for the encoded inputs.
    """

    def __init__(
        self,
        tokenizer,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        device: str = "cuda",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.device = device
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(self, model, prompts) -> Rollouts:
        # The batch flows as example dicts through the rollout sources; extract the prompt string.
        texts = [p["prompt"] if isinstance(p, dict) else p for p in prompts]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        attn_in = enc["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        do_sample = self.temperature > 0.0
        full_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn_in,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response_mask = build_response_mask(full_ids, prompt_len, self.tokenizer.eos_token_id)
        attention_mask = response_mask.clone()
        attention_mask[:, :prompt_len] = attn_in
        return Rollouts(input_ids=full_ids, attention_mask=attention_mask, response_mask=response_mask)
