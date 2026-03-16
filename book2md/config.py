# config.py — Central configuration for the book2md pipeline.
# Edit this file to change models, paths, and generation parameters.

import sys as _sys
from dataclasses import dataclass, field


class _StderrFilter:
    """Suppress known non-fatal protobuf/grpc compatibility tracebacks."""
    _PATTERN = "'MessageFactory' object has no attribute 'GetPrototype'"

    def __init__(self, wrapped): self._wrapped = wrapped
    def write(self, msg):
        if self._PATTERN not in msg:
            self._wrapped.write(msg)
    def flush(self): self._wrapped.flush()
    def __getattr__(self, name): return getattr(self._wrapped, name)

_sys.stderr = _StderrFilter(_sys.stderr)


@dataclass
class PDFConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    dpi: int = 300
    max_new_tokens: int = 4096
    repetition_penalty: float = 1.15
    prompt: str = """Convert this png book page to Markdown. Preserve the original language exactly.

Markdown conventions:
- Headings scheme:  # H1  ## H2  ### H3  #### H4
- Bold: **text**
- Italic: *text*
- Bold+italic: ***text***
- Unordered list: - item
- Ordered list: 1. item
- Table: GFM pipe table with --- separator row
- Image: ![alt](images/filename.ext)
- Footnote: [^n]: text
- Blockquote: > text

Rules:
- Preserve heading hierarchy as in the source
- Use the heading schemes only for titles and section headings
- Replace [IMAGE_N] with ![image_N](images/image_N.png)
- Never duplicate text
- If in the same scanned image there are two pages, separate them according to the page number
- Ignore page numbers (isolated digits at top or bottom of page)
- Ignore running headers/footers (repeated text like author name or chapter title)
- Captions below figures/tables: render as plain text after the image/table, not as headings
- Two-column layouts: merge into single-column reading order (left column first, then right)
- Output ONLY the Markdown, no commentary"""


@dataclass
class EPUBConfig:
    model_id: str = "Qwen/Qwen3-4B"
    max_chunk_chars: int = 8_000
    max_new_tokens: int = 2_048
    repetition_penalty: float = 1.15
    prompt: str = """Convert the following HTML book page to Markdown. Preserve the original language exactly.

Markdown conventions:
- Headings scheme:  # H1  ## H2  ### H3  #### H4
- Bold: **text**
- Italic: *text*
- Bold+italic: ***text***
- Unordered list: - item
- Ordered list: 1. item
- Table: GFM pipe table with --- separator row
- Image: ![alt](images/filename.ext) -- place where the <img> appears in the HTML
- Footnote: [^id]: text
- Blockquote: > text

Rules:
- Preserve heading hierarchy as in the source
- Use the heading schemes only for titles and section headings
- Never duplicate text
- Ignore page numbers (isolated digits)
- Ignore running headers/footers (repeated text like author name or chapter title)
- Captions below figures/tables: render as plain text after the image/table, not as headings
- <aside> and epub:type="footnote" elements: render as [^n]: footnotes
- <cite>: render as *italic*
- Output ONLY the Markdown, no commentary


HTML:
{html}
"""


@dataclass
class MetadataConfig:
    max_new_tokens: int = 128
    biblio_prompt: str = """\
You are a bibliographic information extraction assistant. Extract author, title, and \
publication year from texts in German or Italian.

Output format (JSON):
{"author": "Author Name", "title": "Work Title", "year": "YYYY"}

Rules:
- The user message starts with "Book filename: <name>" — use that as reference for \
author and title if they are not found in the text (fields in the filename are \
separated by _).
- If year is missing, use null.
- Keep original language for author and title.
- Extract only the 4-digit publication year.
- Multiple authors as comma-separated string.
- Output only valid JSON, no additional text.\
"""
    genre_prompt: str = """\
You are a literary genre classifier for Italian and German texts.
Based on the provided excerpt from the body of a book, classify it into one of these genres:
- Journalistic: newspapers, magazines, club publications, press office publications, \
district and municipal gazettes
- Functional/Gebrauchstexte: school books, health guides, cookbooks, hiking guides, \
regional guides, manuals, advertisements, programme booklets, ordinances
- Factual/Non fiction/Wissenschaft: non-fiction, popular science, essays in journals \
or conference proceedings, scientific texts, theses, dissertations, thematic periodicals
- Fiction/Belletristik: novels, novellas, short stories, biographies, literary letters, \
essays, sagas, fairy tales, crime stories, children's literature, drama, autobiographical \
literature, travelogues

Output format (JSON):
{"genre": "Genre"}

Rules:
- Output only valid JSON, no additional text.\
"""


@dataclass
class EvalConfig:
    n: int = 20
    enable_prefix_caching: bool = True


@dataclass
class ParseConfig:
    langs: list = field(default_factory=lambda: ["it", "de"])
    output_format: str = "conllu"  # "conllu", "json", or "both"


@dataclass
class PathConfig:
    input_dir: str = "books/"
    output_dir: str = "output/"
    scores_dir: str = "scores/"
    metadata_csv: str = "metadata/metadata.csv"


# ── Module-level singletons (import and override as needed) ──────────────────
pdf_config      = PDFConfig()
epub_config     = EPUBConfig()
metadata_config = MetadataConfig()
eval_config     = EvalConfig()
parse_config    = ParseConfig()
path_config     = PathConfig()
