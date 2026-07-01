"""Tests for response-mask construction and HFSampler wiring. CPU-only, no downloads."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kd_lab.distillation.sampling import HFSampler, build_response_mask


class TestBuildResponseMask:
    def test_pad_distinct_from_eos(self):
        # prompt_len=2; gen = [5, EOS=9, pad=0, pad=0]; keep up to and including EOS.
        full = torch.tensor([[1, 2, 5, 9, 0, 0]])
        got = build_response_mask(full, prompt_len=2, eos_token_id=9)
        assert torch.equal(got, torch.tensor([[0, 0, 1, 1, 0, 0]]))

    def test_pad_equals_eos(self):
        # pad == eos == 9: only the first EOS is a response token.
        full = torch.tensor([[1, 2, 3, 4, 9, 9, 9]])
        got = build_response_mask(full, prompt_len=2, eos_token_id=9)
        assert torch.equal(got, torch.tensor([[0, 0, 1, 1, 1, 0, 0]]))

    def test_no_eos_keeps_all_generated(self):
        full = torch.tensor([[1, 2, 3, 4, 5]])
        got = build_response_mask(full, prompt_len=2, eos_token_id=9)
        assert torch.equal(got, torch.tensor([[0, 0, 1, 1, 1]]))

    def test_eos_none_keeps_all_generated(self):
        full = torch.tensor([[1, 2, 3, 4, 5]])
        got = build_response_mask(full, prompt_len=2, eos_token_id=None)
        assert torch.equal(got, torch.tensor([[0, 0, 1, 1, 1]]))

    def test_eos_at_first_generated_position(self):
        full = torch.tensor([[1, 2, 9, 0, 0]])
        got = build_response_mask(full, prompt_len=2, eos_token_id=9)
        assert torch.equal(got, torch.tensor([[0, 0, 1, 0, 0]]))

    def test_prompt_len_covers_everything(self):
        full = torch.tensor([[1, 2, 3]])
        got = build_response_mask(full, prompt_len=3, eos_token_id=9)
        assert torch.equal(got, torch.tensor([[0, 0, 0]]))

    def test_batched_rows_independent(self):
        full = torch.tensor([[1, 2, 5, 9, 0], [0, 3, 6, 7, 8]])  # row0 ends early, row1 runs on
        got = build_response_mask(full, prompt_len=2, eos_token_id=9)
        assert torch.equal(got, torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 1]]))


class _StubTokenizer:
    """Mimics the small slice of the HF tokenizer API that HFSampler uses."""

    def __init__(self):
        self.padding_side = "right"
        self.pad_token_id = 0
        self.eos_token_id = 9
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, prompts, return_tensors=None, padding=None):
        # two prompts, left-padded to length 2 (row 1 has a leading pad).
        return {
            "input_ids": torch.tensor([[1, 2], [0, 3]]),
            "attention_mask": torch.tensor([[1, 1], [0, 1]]),
        }


class _StubModel:
    def generate(self, input_ids, attention_mask, **kwargs):
        # append 3 tokens: row0 = [5, EOS, pad], row1 = [6, 7, 8] (no EOS).
        return torch.tensor([[1, 2, 5, 9, 0], [0, 3, 6, 7, 8]])


class TestHFSamplerWiring:
    def test_masks_and_ids(self):
        tok = _StubTokenizer()
        sampler = HFSampler(tok, max_new_tokens=3, temperature=0.0, device="cpu")
        assert tok.padding_side == "left"  # forced for batched causal generation

        roll = sampler.generate(_StubModel(), ["a", "b"])
        assert torch.equal(roll.input_ids, torch.tensor([[1, 2, 5, 9, 0], [0, 3, 6, 7, 8]]))
        # response mask: 0 over the prompt, generated tokens up to and including the first EOS.
        assert torch.equal(roll.response_mask, torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 1]]))
        # attention mask: prompt attention from the tokenizer, generated kept tokens attended.
        assert torch.equal(roll.attention_mask, torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]]))
