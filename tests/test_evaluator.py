"""Tests for QualityEvaluator (no GPU needed)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from book2md.evaluation.evaluator import QualityEvaluator


def make_evaluator(use_bertscore=False) -> QualityEvaluator:
    ev = QualityEvaluator.__new__(QualityEvaluator)
    ev.use_bertscore = use_bertscore
    ev.ned = MagicMock()
    ev.bleu = MagicMock()
    ev.structure_f1 = MagicMock()
    ev.bert = MagicMock() if use_bertscore else None
    ev.ned.score.return_value = 0.1
    ev.bleu.score.return_value = 80.0
    ev.structure_f1.score.return_value = 0.9
    return ev


# ── _build_result ─────────────────────────────────────────────────────────────

class TestBuildResult:
    def test_basic_averaging(self):
        scores = {0: {"ned": 0.1, "bleu": 80.0}, 1: {"ned": 0.3, "bleu": 60.0}}
        result = QualityEvaluator._build_result(scores, ("ned", "bleu"), key="pages")
        assert result["average"]["ned"] == 0.2
        assert result["average"]["bleu"] == 70.0

    def test_none_excluded_from_average(self):
        scores = {0: {"ned": None}, 1: {"ned": 0.4}}
        result = QualityEvaluator._build_result(scores, ("ned",), key="pages")
        assert result["average"]["ned"] == 0.4

    def test_all_none_returns_none_average(self):
        scores = {0: {"ned": None}}
        result = QualityEvaluator._build_result(scores, ("ned",), key="pages")
        assert result["average"]["ned"] is None

    def test_pages_keyed_as_strings(self):
        scores = {0: {"ned": 0.1}, 1: {"ned": 0.2}}
        result = QualityEvaluator._build_result(scores, ("ned",), key="pages")
        assert "0" in result["pages"] and "1" in result["pages"]

    def test_custom_key_name(self):
        scores = {0: {"ned": 0.1}}
        result = QualityEvaluator._build_result(scores, ("ned",), key="chunks")
        assert "chunks" in result and "pages" not in result

    def test_pages_sorted_by_index(self):
        scores = {2: {"ned": 0.3}, 0: {"ned": 0.1}, 1: {"ned": 0.2}}
        result = QualityEvaluator._build_result(scores, ("ned",), key="pages")
        assert list(result["pages"].keys()) == ["0", "1", "2"]

    def test_empty_scores(self):
        result = QualityEvaluator._build_result({}, ("ned",), key="pages")
        assert result["average"]["ned"] is None


# ── _score_pair ───────────────────────────────────────────────────────────────

class TestScorePair:
    def test_returns_all_dims(self):
        ev = make_evaluator()
        result = ev._score_pair("reference text", "predicted text")
        assert set(result.keys()) == {"ned", "bleu", "structure_f1"}

    def test_with_bertscore(self):
        ev = make_evaluator(use_bertscore=True)
        ev.bert.score.return_value = 0.95
        result = ev._score_pair("ref", "pred")
        assert "bertscore" in result

    def test_values_are_rounded(self):
        ev = make_evaluator()
        ev.ned.score.return_value = 0.123456
        result = ev._score_pair("ref", "pred")
        assert result["ned"] == round(0.123456, 4)


# ── _evaluate_dir ─────────────────────────────────────────────────────────────

class TestEvaluateDir:
    def _setup_eval_dir(self, tmp_path, n=2):
        eval_dir = tmp_path / "mybook" / "eval_pages"
        eval_dir.mkdir(parents=True)
        for i in range(n):
            (eval_dir / f"{i}.md").write_text(f"prediction {i}", encoding="utf-8")
            (eval_dir / f"{i}.ref.md").write_text(f"reference {i}", encoding="utf-8")
        return eval_dir

    def test_writes_scores_json(self, tmp_path):
        eval_dir = self._setup_eval_dir(tmp_path)
        scores_dir = tmp_path / "scores"
        ev = make_evaluator()
        ev._evaluate_dir(eval_dir, scores_dir, key="pages", label="page")
        assert (scores_dir / "mybook_scores.json").exists()

    def test_json_contains_average_and_pages(self, tmp_path):
        eval_dir = self._setup_eval_dir(tmp_path)
        scores_dir = tmp_path / "scores"
        ev = make_evaluator()
        result = ev._evaluate_dir(eval_dir, scores_dir, key="pages", label="page")
        assert "average" in result and "pages" in result

    def test_skips_empty_reference(self, tmp_path):
        eval_dir = tmp_path / "mybook" / "eval_pages"
        eval_dir.mkdir(parents=True)
        (eval_dir / "0.md").write_text("prediction", encoding="utf-8")
        (eval_dir / "0.ref.md").write_text("", encoding="utf-8")
        scores_dir = tmp_path / "scores"
        ev = make_evaluator()
        result = ev._evaluate_dir(eval_dir, scores_dir, key="pages", label="page")
        assert result["pages"] == {}

    def test_returns_empty_dict_when_no_refs(self, tmp_path):
        eval_dir = tmp_path / "mybook" / "eval_pages"
        eval_dir.mkdir(parents=True)
        scores_dir = tmp_path / "scores"
        ev = make_evaluator()
        result = ev._evaluate_dir(eval_dir, scores_dir, key="pages", label="page")
        assert result == {}
