"""Tests for MetadataExtractor (no GPU needed)."""
import pandas as pd
import pytest

from book2md.metadata.extractor import MetadataExtractor


# ── _parse_json ───────────────────────────────────────────────────────────────

class TestParseJson:
    def test_valid_json(self):
        raw = '{"author": "Kafka", "title": "Der Prozess", "year": "1925"}'
        assert MetadataExtractor._parse_json(raw) == {"author": "Kafka", "title": "Der Prozess", "year": "1925"}

    def test_strips_surrounding_whitespace(self):
        assert MetadataExtractor._parse_json('  {"genre": "Fiction"}  ') == {"genre": "Fiction"}

    def test_json_code_fence_stripped(self):
        assert MetadataExtractor._parse_json('```json\n{"genre": "Fiction"}\n```') == {"genre": "Fiction"}

    def test_plain_code_fence_stripped(self):
        assert MetadataExtractor._parse_json('```\n{"genre": "Fiction"}\n```') == {"genre": "Fiction"}

    def test_invalid_json_returns_none(self):
        assert MetadataExtractor._parse_json("not json at all!!!") is None

    def test_empty_string_returns_none(self):
        assert MetadataExtractor._parse_json("") is None

    def test_partial_recovery_before_blank_line(self):
        raw = '{"author": "Test"}\n\nextra garbage'
        result = MetadataExtractor._parse_json(raw)
        assert result is not None
        assert result.get("author") == "Test"


# ── collect_samples ───────────────────────────────────────────────────────────

class TestCollectSamples:
    def _extractor(self):
        return MetadataExtractor.__new__(MetadataExtractor)

    def test_finds_eval_pages(self, tmp_path):
        eval_dir = tmp_path / "book1" / "eval_pages"
        eval_dir.mkdir(parents=True)
        for i in range(5):
            (eval_dir / f"{i}.md").write_text(f"page {i}", encoding="utf-8")

        samples = self._extractor().collect_samples(str(tmp_path))
        assert len(samples) == 1
        assert samples[0]["book_name"] == "book1"

    def test_falls_back_to_eval_chunks(self, tmp_path):
        eval_dir = tmp_path / "book2" / "eval_chunks"
        eval_dir.mkdir(parents=True)
        for i in range(3):
            (eval_dir / f"{i}.md").write_text(f"chunk {i}", encoding="utf-8")

        samples = self._extractor().collect_samples(str(tmp_path))
        assert len(samples) == 1
        assert samples[0]["book_name"] == "book2"

    def test_skips_dirs_without_eval_folder(self, tmp_path):
        (tmp_path / "empty_book").mkdir()
        assert self._extractor().collect_samples(str(tmp_path)) == []

    def test_skips_ref_md_files(self, tmp_path):
        """0.ref.md must not be counted as eval page (caused ValueError in stem conversion)."""
        eval_dir = tmp_path / "book" / "eval_pages"
        eval_dir.mkdir(parents=True)
        (eval_dir / "0.md").write_text("page0", encoding="utf-8")
        (eval_dir / "0.ref.md").write_text("ref0", encoding="utf-8")

        samples = self._extractor().collect_samples(str(tmp_path))
        assert len(samples) == 1

    def test_front_text_uses_first_five_pages(self, tmp_path):
        eval_dir = tmp_path / "book" / "eval_pages"
        eval_dir.mkdir(parents=True)
        for i in range(12):
            (eval_dir / f"{i}.md").write_text(f"page{i}", encoding="utf-8")

        samples = self._extractor().collect_samples(str(tmp_path))
        front = samples[0]["front_text"]
        for i in range(5):
            assert f"page{i}" in front
        assert "page5" not in front

    def test_body_text_uses_last_of_guaranteed_10(self, tmp_path):
        eval_dir = tmp_path / "book" / "eval_pages"
        eval_dir.mkdir(parents=True)
        for i in range(12):
            (eval_dir / f"{i}.md").write_text(f"page{i}", encoding="utf-8")

        samples = self._extractor().collect_samples(str(tmp_path))
        body = samples[0]["body_text"]
        assert "page7" in body and "page8" in body and "page9" in body
        assert "page0" not in body

    def test_multiple_books(self, tmp_path):
        for book in ("bookA", "bookB"):
            d = tmp_path / book / "eval_pages"
            d.mkdir(parents=True)
            (d / "0.md").write_text("text", encoding="utf-8")

        samples = self._extractor().collect_samples(str(tmp_path))
        assert {s["book_name"] for s in samples} == {"bookA", "bookB"}


# ── run — CSV resume / append ─────────────────────────────────────────────────

def _make_eval_dir(root, book_name, n=3):
    d = root / book_name / "eval_pages"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i}.md").write_text(f"page{i}", encoding="utf-8")


class TestRunResume:
    def _extractor_with_llm(self,
                             biblio='{"author":"A","title":"T","year":"2000"}',
                             genre='{"genre":"Fiction"}'):
        ex = MetadataExtractor.__new__(MetadataExtractor)
        ex.max_new_tokens = 128
        calls = []

        class FakeLLM:
            def chat(self_, dataset, sp, chat_template_kwargs=None):
                calls.append(len(dataset))
                text = biblio if len(calls) % 2 == 1 else genre
                class Out:
                    def __init__(self, t):
                        self.outputs = [type("O", (), {"text": t})()]
                return [Out(text)] * len(dataset)

        ex.llm = FakeLLM()
        return ex

    def test_no_existing_csv_creates_new(self, tmp_path):
        _make_eval_dir(tmp_path, "bookA")
        csv_path = tmp_path / "meta.csv"
        df = self._extractor_with_llm().run(str(tmp_path), str(csv_path))
        assert csv_path.exists()
        assert len(df) == 1
        assert df.iloc[0]["book"] == "bookA"

    def test_existing_csv_book_is_skipped(self, tmp_path):
        _make_eval_dir(tmp_path, "bookA")
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame([{"book": "bookA", "author": "Old", "title": "Old", "year": "1900", "genre": "Drama"}]).to_csv(csv_path, index=False)
        df = self._extractor_with_llm().run(str(tmp_path), str(csv_path))
        assert df.iloc[0]["author"] == "Old"

    def test_existing_csv_new_book_is_appended(self, tmp_path):
        _make_eval_dir(tmp_path, "bookA")
        _make_eval_dir(tmp_path, "bookB")
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame([{"book": "bookA", "author": "Old", "title": "Old", "year": "1900", "genre": "Drama"}]).to_csv(csv_path, index=False)
        df = self._extractor_with_llm().run(str(tmp_path), str(csv_path))
        assert set(df["book"]) == {"bookA", "bookB"}

    def test_all_books_done_returns_existing_df(self, tmp_path):
        _make_eval_dir(tmp_path, "bookA")
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame([{"book": "bookA", "author": "A", "title": "T", "year": "2000", "genre": "Fiction"}]).to_csv(csv_path, index=False)
        df = self._extractor_with_llm().run(str(tmp_path), str(csv_path))
        assert len(df) == 1

    def test_csv_not_rewritten_when_all_done(self, tmp_path):
        _make_eval_dir(tmp_path, "bookA")
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame([{"book": "bookA", "author": "Orig", "title": "T", "year": "2000", "genre": "Fiction"}]).to_csv(csv_path, index=False)
        mtime_before = csv_path.stat().st_mtime
        self._extractor_with_llm().run(str(tmp_path), str(csv_path))
        assert csv_path.stat().st_mtime == mtime_before
