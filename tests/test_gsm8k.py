"""Tests for GSM8K parsing/bucketing/scoring. Pure helpers, no dataset download."""

from __future__ import annotations

from kd_lab.tasks.gsm8k import (
    eval_sets_by_length,
    extract_gold_answer,
    length_bucket,
    make_example,
    num_reasoning_steps,
    parse_pred_answer,
    score_gsm8k,
)

_ANSWER = "Step one costs 2.\nStep two adds 3 more.\nSo total is 5.\n#### 5"


def test_extract_gold_answer_with_commas():
    assert extract_gold_answer("blah\n#### 1,234") == 1234
    assert extract_gold_answer("#### -7") == -7


def test_num_reasoning_steps():
    assert num_reasoning_steps(_ANSWER) == 3


def test_length_bucket_monotone():
    edges = (2, 4, 6)
    assert length_bucket(1, edges) == 0
    assert length_bucket(2, edges) == 0
    assert length_bucket(3, edges) == 1
    assert length_bucket(7, edges) == 3  # beyond the last edge


def test_make_example_shape_and_target():
    ex = make_example("What is 2+3?", _ANSWER)
    assert ex["answer"] == 5
    assert ex["target"].endswith("Final: 5")
    assert "Problem: What is 2+3?" in ex["prompt"]
    assert ex["n_steps"] == 3 and ex["k"] == length_bucket(3)


def test_parse_pred_answer_prefers_final():
    assert parse_pred_answer("reasoning ... Final: 42") == 42
    assert parse_pred_answer("no marker, ends with 17") == 17
    assert parse_pred_answer("with commas Final: 1,000") == 1000
    assert parse_pred_answer("no digits here") is None


def test_score_gsm8k():
    ex = make_example("q", _ANSWER)  # answer 5
    assert score_gsm8k("the answer is Final: 5", ex) is True
    assert score_gsm8k("Final: 6", ex) is False


def test_eval_sets_by_length_groups_and_sorts():
    exs = [
        {"k": 2, "answer": 1},
        {"k": 0, "answer": 2},
        {"k": 2, "answer": 3},
    ]
    sets = eval_sets_by_length(exs)
    assert list(sets.keys()) == [0, 2]
    assert len(sets[2]) == 2
