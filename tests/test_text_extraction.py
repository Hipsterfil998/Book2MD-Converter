"""Tests for DocumentProcessor (text_extraction.py) — no GPU needed."""
import pytest
from bs4 import BeautifulSoup

from converters.text_extraction import DocumentProcessor

proc = DocumentProcessor()


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_block(spans: list[tuple], bbox_y: float = 0.0) -> dict:
    """Build a PyMuPDF rawdict text block from (text, flags, size) triples."""
    return {
        "bbox": [0, bbox_y, 100, bbox_y + 20],
        "lines": [{"spans": [{"text": t, "flags": f, "size": s} for t, f, s in spans]}],
    }


def parse_el(html: str):
    return BeautifulSoup(html, "html.parser").find()


# ── PDF block → Markdown ───────────────────────────────────────────────────────

class TestPdfBlockToMarkdown:
    def test_plain_text(self):
        assert proc._pdf_block_to_markdown(make_block([("Hello world", 0, 11)])) == "Hello world"

    def test_bold(self):
        assert proc._pdf_block_to_markdown(make_block([("bold", 16, 11)])) == "**bold**"

    def test_italic(self):
        assert proc._pdf_block_to_markdown(make_block([("ital", 2, 11)])) == "*ital*"

    def test_bold_italic(self):
        assert proc._pdf_block_to_markdown(make_block([("bi", 18, 11)])) == "***bi***"

    def test_heading_h1(self):
        assert proc._pdf_block_to_markdown(make_block([("Title", 0, 24)])) == "# Title"

    def test_heading_h2(self):
        assert proc._pdf_block_to_markdown(make_block([("Section", 0, 19)])) == "## Section"

    def test_heading_h3(self):
        assert proc._pdf_block_to_markdown(make_block([("Sub", 0, 15)])) == "### Sub"

    def test_footnote(self):
        result = proc._pdf_block_to_markdown(make_block([("1 Footnote text", 0, 8)]))
        assert result.startswith("> [^fn]:")

    def test_caption_figure(self):
        result = proc._pdf_block_to_markdown(make_block([("Figure 1: A chart", 0, 11)]))
        assert result.startswith("*") and result.endswith("*")

    def test_empty_block(self):
        assert proc._pdf_block_to_markdown({"bbox": [0, 0, 100, 20], "lines": []}) == ""


# ── Caption detection ──────────────────────────────────────────────────────────

class TestFindPdfCaption:
    def test_finds_caption_just_below_image(self):
        blocks = [(110.0, "Figure 1: A chart")]
        caption = DocumentProcessor._find_pdf_caption(blocks, img_y0=100.0, img_height=50.0)
        assert caption is not None
        assert "Figure 1" in caption

    def test_no_caption_when_too_far(self):
        blocks = [(200.0, "Figure 1: A chart")]
        assert DocumentProcessor._find_pdf_caption(blocks, img_y0=100.0, img_height=50.0) is None

    def test_no_caption_when_pattern_unmatched(self):
        blocks = [(110.0, "Regular text below image")]
        assert DocumentProcessor._find_pdf_caption(blocks, img_y0=100.0, img_height=50.0) is None


# ── GFM table ─────────────────────────────────────────────────────────────────

class TestTableToMarkdown:
    def test_basic_table(self):
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        result = DocumentProcessor._table_to_markdown(parse_el(html))
        assert "| A | B |" in result
        assert "| --- | --- |" in result
        assert "| 1 | 2 |" in result

    def test_empty_table(self):
        assert DocumentProcessor._table_to_markdown(parse_el("<table></table>")) == ""

    def test_header_only(self):
        html = "<table><tr><th>Col1</th><th>Col2</th></tr></table>"
        result = DocumentProcessor._table_to_markdown(parse_el(html))
        assert "| Col1 | Col2 |" in result
        assert "---" in result


# ── HTML element → Markdown ───────────────────────────────────────────────────

class TestElementToMarkdown:
    def test_h1(self):
        assert "# Title" in DocumentProcessor._element_to_markdown(parse_el("<h1>Title</h1>"))

    def test_h3(self):
        assert "### Sub" in DocumentProcessor._element_to_markdown(parse_el("<h3>Sub</h3>"))

    def test_bold(self):
        assert DocumentProcessor._element_to_markdown(parse_el("<strong>Bold</strong>")) == "**Bold**"

    def test_italic(self):
        assert DocumentProcessor._element_to_markdown(parse_el("<em>Italic</em>")) == "*Italic*"

    def test_link(self):
        result = DocumentProcessor._element_to_markdown(
            parse_el('<a href="https://example.com">Click</a>')
        )
        assert result == "[Click](https://example.com)"

    def test_link_without_href_returns_inner_text(self):
        assert DocumentProcessor._element_to_markdown(parse_el("<a>text</a>")) == "text"

    def test_hr(self):
        assert "---" in DocumentProcessor._element_to_markdown(parse_el("<hr/>"))

    def test_image(self):
        result = DocumentProcessor._element_to_markdown(parse_el('<img src="fig.png" alt="desc"/>'))
        assert "![desc](fig.png)" in result

    def test_unordered_list(self):
        result = DocumentProcessor._element_to_markdown(
            parse_el("<ul><li>A</li><li>B</li></ul>")
        )
        assert "- A" in result
        assert "- B" in result

    def test_ordered_list(self):
        result = DocumentProcessor._element_to_markdown(
            parse_el("<ol><li>First</li><li>Second</li></ol>")
        )
        assert "1. First" in result
        assert "2. Second" in result

    def test_paragraph(self):
        assert "Hello" in DocumentProcessor._element_to_markdown(parse_el("<p>Hello</p>"))

    def test_inline_code(self):
        result = DocumentProcessor._element_to_markdown(
            parse_el("<p><code>fn()</code></p>")
        )
        assert "`fn()`" in result


# ── EPUB HTML → Markdown ──────────────────────────────────────────────────────

class TestEpubHtmlToMarkdown:
    def test_strips_nav_element(self):
        html = "<html><body><nav>skip me</nav><p>Content</p></body></html>"
        result = proc._epub_html_to_markdown(html)
        assert "skip me" not in result
        assert "Content" in result

    def test_converts_headings(self):
        html = "<html><body><h2>Chapter</h2><p>Text</p></body></html>"
        result = proc._epub_html_to_markdown(html)
        assert "## Chapter" in result

    def test_converts_table(self):
        html = (
            "<html><body>"
            "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
            "</body></html>"
        )
        result = proc._epub_html_to_markdown(html)
        assert "| A |" in result
        assert "---" in result
