"""Offline validation of the experiment runner orchestration.

Uses tiny torch models + a word-level stub tokenizer + a tiled sampler injected through
``RunComponents``, so the whole ``run_condition`` train-loop-and-eval runs on CPU with no
HuggingFace download and no GPU. This validates the control flow, method routing, the off-policy
encoder, decode, the eval, and the results file, not any scientific number.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kd_lab.distillation.on_policy import Rollouts, Sampler
from kd_lab.experiments.run import (
    EncodedRolloutDataset,
    RunComponents,
    build_optimizer,
    decode_responses,
    run_condition,
)

VOCAB = 8000  # comfortably larger than the tiny stub vocab a small dataset produces


class _ToyLM(torch.nn.Module):
    def __init__(self, vocab: int = VOCAB, dim: int = 8) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))


class _StubTokenizer:
    """Growing word-level tokenizer: encode/decode round-trip, bounded ids for a tiny dataset."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._vocab = {"<pad>": 0, "<eos>": 1}
        self._inv = {0: "<pad>", 1: "<eos>"}

    def _id(self, w: str) -> int:
        if w not in self._vocab:
            i = len(self._vocab)
            self._vocab[w] = i
            self._inv[i] = w
        return self._vocab[w]

    def __call__(self, text, add_special_tokens=False, return_tensors=None, padding=None):
        return {"input_ids": [self._id(w) for w in text.split()]}

    def decode(self, ids) -> str:
        skip = {self.pad_token_id, self.eos_token_id}
        return " ".join(self._inv.get(int(i), "<unk>") for i in ids if int(i) not in skip)


class _TiledSampler(Sampler):
    """Returns a batch of identical rollouts whose response decodes to a fixed string."""

    def __init__(self, tokenizer, response_text: str, prompt_len: int = 2) -> None:
        self.tok = tokenizer
        self.text = response_text
        self.prompt_len = prompt_len

    def generate(self, model, prompts) -> Rollouts:
        b = len(prompts)
        resp = self.tok(self.text)["input_ids"] + [self.tok.eos_token_id]
        row = [self.tok.pad_token_id] * self.prompt_len + resp
        t = len(row)
        input_ids = torch.tensor([row] * b, dtype=torch.long)
        attention_mask = torch.ones((b, t), dtype=torch.long)
        response_mask = torch.zeros((b, t), dtype=torch.long)
        response_mask[:, self.prompt_len :] = 1
        return Rollouts(input_ids=input_ids, attention_mask=attention_mask, response_mask=response_mask)


def _base_cfg(method: str, divergence: str, run_name: str) -> dict:
    return {
        "run_name": run_name,
        "seed": 0,
        "device": "cpu",
        "task": {
            "num_nodes": 8,
            "shuffle_table": True,
            "train_k": [1, 2],
            "eval_k": [1, 2, 3],
            "n_train": 16,
            "n_eval_per_k": 3,
        },
        "distillation": {
            "method": method,
            "divergence": divergence,
            "temperature": 1.0,
            "on_policy_fraction": 1.0,
        },
        "optim": {
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "max_steps": 3,
            "batch_size": 4,
            "grad_clip": 1.0,
            "max_seq_len": 128,
        },
        "eval": {"probe_examples": 3},
    }


def _components(cfg: dict, tokenizer) -> RunComponents:
    student, teacher = _ToyLM(), _ToyLM()
    optimizer, scheduler = build_optimizer(cfg, student)
    sampler = _TiledSampler(tokenizer, "Final: 3")
    off_dataset = EncodedRolloutDataset(
        tokenizer, max_len=128, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
    )
    return RunComponents(
        student=student,
        teacher=teacher,
        tokenizer=tokenizer,
        sampler=sampler,
        eval_sampler=sampler,
        off_dataset=off_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
    )


class TestBuildOptimizer:
    def test_cosine_warmup_shape(self):
        cfg = _base_cfg("sft", "forward_kl", "opt")
        cfg["optim"]["warmup_steps"] = 5
        cfg["optim"]["max_steps"] = 20
        student = _ToyLM()
        optimizer, scheduler = build_optimizer(cfg, student)
        base = cfg["optim"]["lr"]
        lrs = []
        for _ in range(20):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        assert lrs[0] < base  # warmup starts below base lr
        assert lrs[5] == pytest.approx(base, rel=1e-6)  # base lr reached at end of warmup
        assert lrs[-1] < lrs[5]  # cosine decay afterwards


class TestEncoderAndDecode:
    def test_encode_batch_and_decode_roundtrip(self):
        tok = _StubTokenizer()
        ds = EncodedRolloutDataset(tok, max_len=64, eos_token_id=tok.eos_token_id, pad_token_id=0)
        examples = [
            {"prompt": "start here", "target": "3 -> 7 Final: 7"},
            {"prompt": "go", "target": "Final: 2"},
        ]
        roll = ds.batch(examples)
        assert roll.input_ids.shape == roll.response_mask.shape == roll.attention_mask.shape
        # response region decodes back to the target text.
        decoded = decode_responses(tok, roll)
        assert decoded[0] == "3 -> 7 Final: 7"
        assert decoded[1] == "Final: 2"
        # prompt tokens are excluded from the response mask.
        assert int(roll.response_mask[0, 0]) == 0


class TestRunCondition:
    def test_sft_end_to_end(self, tmp_path):
        cfg = _base_cfg("sft", "forward_kl", "test_sft")
        comp = _components(cfg, _StubTokenizer())
        results = run_condition(cfg, comp, results_root=str(tmp_path))

        assert results["plan"]["method"] == "sft"
        assert set(results["horizon_accuracy"]) == {1, 2, 3}
        for stats in results["horizon_accuracy"].values():
            assert 0.0 <= stats["mean"] <= 1.0 and stats["n"] == 3
        assert len(results["positional_kl"]["mean_kl"]) == len(results["positional_kl"]["position"])
        assert (tmp_path / "test_sft" / "metrics.json").exists()

    def test_on_policy_end_to_end(self, tmp_path):
        cfg = _base_cfg("on_policy", "reverse_kl", "test_opd")
        comp = _components(cfg, _StubTokenizer())
        results = run_condition(cfg, comp, results_root=str(tmp_path))

        assert results["plan"]["method"] == "on_policy"
        assert results["plan"]["divergence_name"] == "reverse_kl"
        assert len(results["train_log"]) >= 1
        assert all("loss" in e for e in results["train_log"])
        assert (tmp_path / "test_opd" / "metrics.json").exists()
