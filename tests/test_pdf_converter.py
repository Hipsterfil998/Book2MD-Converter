"""Tests for PDFToMarkdownConverter (static methods only — no GPU needed)."""
from converters.pdf2md_LLM import PDFToMarkdownConverter


class TestClean:
    def test_strips_markdown_fence(self):
        text = "```markdown\n# Title\nContent\n```"
        assert PDFToMarkdownConverter._clean(text) == "# Title\nContent"

    def test_strips_plain_fence(self):
        text = "```\n# Title\n```"
        assert PDFToMarkdownConverter._clean(text) == "# Title"

    def test_no_fence_unchanged(self):
        assert PDFToMarkdownConverter._clean("# Title\nContent") == "# Title\nContent"

    def test_strips_surrounding_whitespace(self):
        assert PDFToMarkdownConverter._clean("  # Title  ") == "# Title"

    def test_unclosed_fence_still_stripped(self):
        """Opening fence without closing fence: only first line removed."""
        text = "```markdown\n# Title\nContent"
        assert PDFToMarkdownConverter._clean(text) == "# Title\nContent"


class TestResolveImageRefs:
    def test_replaces_placeholder(self):
        result = PDFToMarkdownConverter._resolve_image_refs(
            "Text [IMAGE_1] end", ["fig1.png"]
        )
        assert "![image_1](images/fig1.png)" in result
        assert "[IMAGE_1]" not in result

    def test_appends_when_placeholder_missing(self):
        result = PDFToMarkdownConverter._resolve_image_refs(
            "No placeholder here", ["fig1.png"]
        )
        assert "![image_1](images/fig1.png)" in result
        assert "No placeholder here" in result

    def test_no_images_leaves_text_unchanged(self):
        assert PDFToMarkdownConverter._resolve_image_refs("Some text", []) == "Some text"

    def test_multiple_images_replaced_in_order(self):
        result = PDFToMarkdownConverter._resolve_image_refs(
            "[IMAGE_1] text [IMAGE_2]", ["a.png", "b.png"]
        )
        assert "![image_1](images/a.png)" in result
        assert "![image_2](images/b.png)" in result

    def test_missing_placeholder_appended_at_end(self):
        """[IMAGE_1] replaced inline; [IMAGE_2] absent → appended."""
        result = PDFToMarkdownConverter._resolve_image_refs(
            "[IMAGE_1] text", ["a.png", "b.png"]
        )
        assert "![image_1](images/a.png)" in result
        assert "![image_2](images/b.png)" in result
