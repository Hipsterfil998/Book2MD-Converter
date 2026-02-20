import re
from dataclasses import dataclass, asdict
from pathlib import Path
import ebooklib
import fitz
from ebooklib import epub
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _SourceMetrics:
    pages: int
    images: int
    tables: int
    footnotes: int
    captions: int
    headings: int


@dataclass
class _MarkdownMetrics:
    images: int
    tables: int
    footnotes: int
    captions: int
    headings: int
    heading_hierarchy_ok: bool
    garbage_chars: int        # encoding artifacts
    empty_lines_ratio: float  # proxy for whitespace noise


@dataclass
class QualityReport:
    source_file: str
    markdown_file: str
    source: _SourceMetrics
    markdown: _MarkdownMetrics
    preservation_scores: dict
    overall_score: float


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class QualityAnalyzer:
    """Compare a source document (PDF/EPUB) against its extracted markdown."""

    _HEADING_RE = re.compile(r"^(#{1,6})\s+.+", re.MULTILINE)
    _IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
    _TABLE_ROW_RE = re.compile(r"^\|.+\|$")
    _FOOTNOTE_DEF_RE = re.compile(r"^\[\^[^\]]+\]:", re.MULTILINE)
    _FOOTNOTE_REF_RE = re.compile(r"\[\^\w+\]")
    _CAPTION_RE = re.compile(r"^\*?(figura|figure|fig\.?|tabella|table|tab\.?)\s*\d*", re.IGNORECASE | re.MULTILINE)
    _GARBAGE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")

    def analyze(self, source_path: str, markdown_path: str) -> dict:
        """
        Run quality analysis and return a JSON-serializable report.

        Args:
            source_path:   Path to the original PDF or EPUB.
            markdown_path: Path to the extracted markdown file.
        """
        ext = Path(source_path).suffix.lower()
        if ext == ".pdf":
            src = self._metrics_from_pdf(source_path)
        elif ext == ".epub":
            src = self._metrics_from_epub(source_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        md = self._metrics_from_markdown(markdown_path)
        scores, overall = self._compute_scores(src, md)

        return asdict(QualityReport(
            source_file=source_path,
            markdown_file=markdown_path,
            source=src,
            markdown=md,
            preservation_scores=scores,
            overall_score=overall,
        ))

    # --- Source parsers -------------------------------------------------------

    @staticmethod
    def _metrics_from_pdf(path: str) -> _SourceMetrics:
        doc = fitz.open(path)
        images, tables, footnotes, captions, headings = 0, 0, 0, 0, 0

        caption_re = re.compile(r"^(figura|figure|fig\.?|tabella|table|tab\.?)\s*\d*", re.I)
        footnote_re = re.compile(r"^[\d\*†‡§]")

        for page in doc:
            images += len(page.get_images(full=False))
            try:
                tables += len(page.find_tables().tables)
            except AttributeError:
                pass

            for block in page.get_text("rawdict")["blocks"]:
                if block.get("type") != 0:
                    continue
                sizes = [s.get("size", 0) for l in block["lines"] for s in l["spans"]]
                flags = [s.get("flags", 0) for l in block["lines"] for s in l["spans"]]
                text = " ".join(s.get("text", "") for l in block["lines"] for s in l["spans"]).strip()
                max_size = max(sizes) if sizes else 0

                if max_size >= 12 and (max_size >= 14 or any(f & 16 for f in flags)):
                    headings += 1
                if caption_re.match(text):
                    captions += 1
                if max_size < 9 and footnote_re.match(text):
                    footnotes += 1

        return _SourceMetrics(pages=len(doc), images=images, tables=tables,
                              footnotes=footnotes, captions=captions, headings=headings)

    @staticmethod
    def _metrics_from_epub(path: str) -> _SourceMetrics:
        book = epub.read_epub(path, options={"ignore_ncx": True})
        images, tables, footnotes, captions, headings, pages = 0, 0, 0, 0, 0, 0

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            pages += 1
            soup = BeautifulSoup(item.get_content(), "html.parser")
            images += len(soup.find_all("img"))
            tables += len(soup.find_all("table"))
            headings += sum(len(soup.find_all(h)) for h in ("h1", "h2", "h3", "h4", "h5", "h6"))
            captions += len(soup.find_all("figcaption"))
            captions += len(soup.find_all(class_=re.compile("caption", re.I)))
            footnotes += len(soup.find_all(attrs={"epub:type": "footnote"}))
            footnotes += len(soup.find_all(class_=re.compile(r"footnote|endnote", re.I)))

        return _SourceMetrics(pages=pages, images=images, tables=tables,
                              footnotes=footnotes, captions=captions, headings=headings)

    # --- Markdown parser ------------------------------------------------------

    def _metrics_from_markdown(self, path: str) -> _MarkdownMetrics:
        content = Path(path).read_text(encoding="utf-8")
        lines = content.splitlines()

        heading_matches = self._HEADING_RE.findall(content)
        heading_levels = [len(h) for h in heading_matches]
        hierarchy_ok = all(b - a <= 1 for a, b in zip(heading_levels, heading_levels[1:]))

        # Count table blocks (not individual rows)
        in_table = False
        table_count = 0
        for line in lines:
            is_row = bool(self._TABLE_ROW_RE.match(line.strip()))
            if is_row and not in_table:
                table_count += 1
            in_table = is_row

        fn_defs = self._FOOTNOTE_DEF_RE.findall(content)
        footnotes = len(fn_defs) if fn_defs else len(set(self._FOOTNOTE_REF_RE.findall(content)))
        empty_lines = sum(1 for l in lines if not l.strip())

        return _MarkdownMetrics(
            images=len(self._IMAGE_RE.findall(content)),
            tables=table_count,
            footnotes=footnotes,
            captions=len(self._CAPTION_RE.findall(content)),
            headings=len(heading_matches),
            heading_hierarchy_ok=hierarchy_ok,
            garbage_chars=len(self._GARBAGE_RE.findall(content)),
            empty_lines_ratio=round(empty_lines / max(len(lines), 1), 3),
        )

    # --- Scoring --------------------------------------------------------------

    @staticmethod
    def _preservation_score(source_val: int, md_val: int) -> float:
        """Ratio in [0, 1]; penalizes both under-extraction and hallucination."""
        if source_val == 0:
            return 1.0 if md_val == 0 else 0.0
        return round(min(md_val / source_val, 1.0), 3)

    def _compute_scores(self, src: _SourceMetrics, md: _MarkdownMetrics) -> tuple[dict, float]:
        fields = ["images", "tables", "footnotes", "captions", "headings"]
        scores = {f: self._preservation_score(getattr(src, f), getattr(md, f)) for f in fields}
        return scores, round(sum(scores.values()) / len(scores), 3)

