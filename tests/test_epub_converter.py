"""Tests for EpubToMarkdownConverter (no GPU needed)."""
from pathlib import Path

from book2md.converters.epub import EpubToMarkdownConverter


def make_converter(max_chunk_chars: int = 500) -> EpubToMarkdownConverter:
    c = EpubToMarkdownConverter.__new__(EpubToMarkdownConverter)
    c.max_chunk_chars = max_chunk_chars
    return c


class TestChunk:
    def test_single_block_not_split(self):
        chunks = make_converter(1000)._chunk("<div>Hello world</div>")
        assert len(chunks) == 1

    def test_splits_when_exceeds_limit(self):
        html = "<div>" + "a" * 40 + "</div><div>" + "b" * 40 + "</div>"
        chunks = make_converter(50)._chunk(html)
        assert len(chunks) == 2

    def test_empty_string_returns_no_chunks(self):
        assert make_converter()._chunk("") == []

    def test_all_content_preserved(self):
        html = "<div>alpha</div><div>beta</div><div>gamma</div>"
        joined = "".join(make_converter(50)._chunk(html))
        assert "alpha" in joined and "beta" in joined and "gamma" in joined


class TestRewriteImgSrcs:
    def test_replaces_matching_src(self):
        html = '<html><body><img src="OEBPS/images/fig1.png"/></body></html>'
        mapping = {"OEBPS/images/fig1.png": Path("/local/fig1.png")}
        assert "/local/fig1.png" in make_converter()._rewrite_img_srcs(html, mapping)

    def test_unmatched_src_left_unchanged(self):
        html = '<html><body><img src="other.png"/></body></html>'
        mapping = {"OEBPS/images/fig1.png": Path("/local/fig1.png")}
        result = make_converter()._rewrite_img_srcs(html, mapping)
        assert "other.png" in result
        assert "/local/fig1.png" not in result

    def test_empty_mapping_leaves_html_unchanged(self):
        html = '<html><body><img src="fig.png"/></body></html>'
        assert "fig.png" in make_converter()._rewrite_img_srcs(html, {})
