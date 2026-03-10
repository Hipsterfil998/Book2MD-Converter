"""Tests for QualityEvaluator (no GPU needed)."""
import json
from unittest.mock import MagicMock

import pytest

from quality_evaluation.evaluator import QualityEvaluator


def make_evaluator() -> QualityEvaluator:
    return QualityEvaluator.__new__(QualityEvaluator)


def make_llm_output(text: str) -> MagicMock:
    out = MagicMock()
    out.outputs[0].text = text
    return out


# ── _build_result ─────────────────────────────────────────────────────────────

class TestBuildResult:
    def test_basic_averaging(self):
        scores = {0: {"text": 4, "structure": 2}, 1: {"text": 2, "structure": 4}}
        result = QualityEvaluator._build_result(scores, ("text", "structure"), key="pages")
        assert result["average"]["text"] == 3.0
        assert result["average"]["structure"] == 3.0

    def test_none_excluded_from_sum(self):
        """None values are skipped in the sum; denominator is still len(scores)."""
        scores = {0: {"text": None, "structure": 4}, 1: {"text": 4, "structure": 4}}
        result = QualityEvaluator._build_result(scores, ("text", "structure"), key="pages")
        # text: sum=4, count=2 → 4/2 = 2.0
        assert result["average"]["text"] == 2.0
        assert result["average"]["structure"] == 4.0

    def test_pages_keyed_as_strings(self):
        scores = {0: {"text": 3}, 1: {"text": 5}}
        result = QualityEvaluator._build_result(scores, ("text",), key="pages")
        assert "0" in result["pages"]
        assert "1" in result["pages"]

    def test_custom_key_name(self):
        scores = {0: {"text": 3}}
        result = QualityEvaluator._build_result(scores, ("text",), key="chunks")
        assert "chunks" in result
        assert "pages" not in result

    def test_pages_sorted_by_index(self):
        scores = {2: {"text": 3}, 0: {"text": 5}, 1: {"text": 4}}
        result = QualityEvaluator._build_result(scores, ("text",), key="pages")
        assert list(result["pages"].keys()) == ["0", "1", "2"]

    def test_three_dimensions(self):
        scores = {0: {"text": 3, "structure": 4, "math": 5}}
        result = QualityEvaluator._build_result(
            scores, ("text", "structure", "math"), key="pages"
        )
        assert set(result["average"].keys()) == {"text", "structure", "math"}


# ── _collect_and_save ─────────────────────────────────────────────────────────

class TestCollectAndSave:
    def test_writes_scores_json(self, tmp_path):
        eval_dir = tmp_path / "mybook" / "eval_pages"
        eval_dir.mkdir(parents=True)
        f0 = eval_dir / "0.png"; f0.touch()
        f1 = eval_dir / "1.png"; f1.touch()

        outputs = [
            make_llm_output('{"text": 4, "structure": 3, "math": 2}'),
            make_llm_output('{"text": 2, "structure": 5, "math": 1}'),
        ]
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()

        result = make_evaluator()._collect_and_save(
            [f0, f1], outputs,
            fallback={"text": None, "structure": None, "math": None},
            dims=("text", "structure", "math"),
            key="pages",
            eval_dir=eval_dir,
            scores_dir=scores_dir,
        )

        json_file = scores_dir / "mybook_scores.json"
        assert json_file.exists()
        saved = json.loads(json_file.read_text())
        assert "average" in saved
        assert "pages" in saved

    def test_averages_correct(self, tmp_path):
        eval_dir = tmp_path / "book" / "eval_pages"
        eval_dir.mkdir(parents=True)
        f0 = eval_dir / "0.png"; f0.touch()
        f1 = eval_dir / "1.png"; f1.touch()
        scores_dir = tmp_path / "scores2"
        scores_dir.mkdir()

        result = make_evaluator()._collect_and_save(
            [f0, f1],
            [make_llm_output('{"text": 4, "structure": 3, "math": 2}'),
             make_llm_output('{"text": 2, "structure": 5, "math": 1}')],
            fallback={"text": None, "structure": None, "math": None},
            dims=("text", "structure", "math"),
            key="pages",
            eval_dir=eval_dir,
            scores_dir=scores_dir,
        )
        # text=(4+2)/2=3.0, structure=(3+5)/2=4.0, math=(2+1)/2=1.5
        assert result["average"]["text"] == 3.0
        assert result["average"]["structure"] == 4.0
        assert result["average"]["math"] == 1.5

    def test_fallback_on_invalid_json(self, tmp_path):
        eval_dir = tmp_path / "book" / "eval_pages"
        eval_dir.mkdir(parents=True)
        f0 = eval_dir / "0.png"; f0.touch()
        scores_dir = tmp_path / "scores3"
        scores_dir.mkdir()

        fallback = {"text": None, "structure": None, "math": None}
        result = make_evaluator()._collect_and_save(
            [f0], [make_llm_output("not valid json")],
            fallback=fallback,
            dims=("text", "structure", "math"),
            key="pages",
            eval_dir=eval_dir,
            scores_dir=scores_dir,
        )
        assert result["pages"]["0"] == fallback
