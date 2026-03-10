"""Tests for MetadataExtractor (no GPU needed)."""
import pytest

from metadata.metadata_extractor import MetadataExtractor


# ── _parse_json ───────────────────────────────────────────────────────────────

class TestParseJson:
    def test_valid_json(self):
        raw = '{"author": "Kafka", "title": "Der Prozess", "year": "1925"}'
        result = MetadataExtractor._parse_json(raw)
        assert result == {"author": "Kafka", "title": "Der Prozess", "year": "1925"}

    def test_strips_surrounding_whitespace(self):
        assert MetadataExtractor._parse_json('  {"genre": "Fiction"}  ') == {"genre": "Fiction"}

    def test_json_code_fence_stripped(self):
        raw = '```json\n{"genre": "Fiction"}\n```'
        assert MetadataExtractor._parse_json(raw) == {"genre": "Fiction"}

    def test_plain_code_fence_stripped(self):
        raw = '```\n{"genre": "Fiction"}\n```'
        assert MetadataExtractor._parse_json(raw) == {"genre": "Fiction"}

    def test_invalid_json_returns_none(self):
        assert MetadataExtractor._parse_json("not json at all!!!") is None

    def test_empty_string_returns_none(self):
        assert MetadataExtractor._parse_json("") is None

    def test_partial_recovery_before_blank_line(self):
        """First JSON object before a blank line should be recovered."""
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

    def test_skips_files_at_top_level(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not a book")
        assert self._extractor().collect_samples(str(tmp_path)) == []

    def test_front_text_uses_first_three_pages(self, tmp_path):
        eval_dir = tmp_path / "book" / "eval_pages"
        eval_dir.mkdir(parents=True)
        for i in range(6):
            (eval_dir / f"{i}.md").write_text(f"page{i}", encoding="utf-8")

        samples = self._extractor().collect_samples(str(tmp_path))
        front = samples[0]["front_text"]
        assert "page0" in front
        assert "page1" in front
        assert "page2" in front
        assert "page3" not in front

    def test_body_text_uses_middle_pages(self, tmp_path):
        eval_dir = tmp_path / "book" / "eval_pages"
        eval_dir.mkdir(parents=True)
        for i in range(8):
            (eval_dir / f"{i}.md").write_text(f"page{i}", encoding="utf-8")

        samples = self._extractor().collect_samples(str(tmp_path))
        body = samples[0]["body_text"]
        # body_files = md_files[3:-2] = [3,4,5], body_text = first 2 = [3,4]
        assert "page3" in body or "page4" in body

    def test_multiple_books(self, tmp_path):
        for book in ("bookA", "bookB"):
            d = tmp_path / book / "eval_pages"
            d.mkdir(parents=True)
            (d / "0.md").write_text("text", encoding="utf-8")

        samples = self._extractor().collect_samples(str(tmp_path))
        assert len(samples) == 2
        names = {s["book_name"] for s in samples}
        assert names == {"bookA", "bookB"}
