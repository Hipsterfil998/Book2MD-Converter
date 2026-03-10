"""Tests for EpubToMarkdownConverter (no GPU needed)."""
from pathlib import Path

from converters.epub2md_LLM import EpubToMarkdownConverter


def make_converter(max_chunk_chars: int = 500) -> EpubToMarkdownConverter:
    """Instantiate without calling __init__ (which loads the LLM)."""
    c = EpubToMarkdownConverter.__new__(EpubToMarkdownConverter)
    c.max_chunk_chars = max_chunk_chars
    return c


class TestChunk:
    def test_single_block_not_split(self):
        c = make_converter(1000)
        chunks = c._chunk("<div>Hello world</div>")
        assert len(chunks) == 1

    def test_splits_when_exceeds_limit(self):
        c = make_converter(50)
        html = "<div>" + "a" * 40 + "</div><div>" + "b" * 40 + "</div>"
        chunks = c._chunk(html)
        assert len(chunks) == 2

    def test_empty_string_returns_no_chunks(self):
        c = make_converter()
        assert c._chunk("") == []

    def test_all_content_preserved(self):
        c = make_converter(50)
        html = "<div>alpha</div><div>beta</div><div>gamma</div>"
        joined = "".join(c._chunk(html))
        assert "alpha" in joined
        assert "beta" in joined
        assert "gamma" in joined


class TestRewriteImgSrcs:
    def test_replaces_matching_src(self):
        c = make_converter()
        html = '<html><body><img src="OEBPS/images/fig1.png"/></body></html>'
        mapping = {"OEBPS/images/fig1.png": Path("/local/fig1.png")}
        result = c._rewrite_img_srcs(html, mapping)
        assert "/local/fig1.png" in result

    def test_unmatched_src_left_unchanged(self):
        c = make_converter()
        html = '<html><body><img src="other.png"/></body></html>'
        mapping = {"OEBPS/images/fig1.png": Path("/local/fig1.png")}
        result = c._rewrite_img_srcs(html, mapping)
        assert "other.png" in result
        assert "/local/fig1.png" not in result

    def test_empty_mapping_leaves_html_unchanged(self):
        c = make_converter()
        html = '<html><body><img src="fig.png"/></body></html>'
        result = c._rewrite_img_srcs(html, {})
        assert "fig.png" in result
