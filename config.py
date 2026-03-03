# config.py — Central configuration for the book conversion pipeline
# Edit this file to change models, paths, and generation parameters.

# ── Models ───────────────────────────────────────────────────────────────────
PDF_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"  # vision-language (PDF → MD, PDF eval)
TEXT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"     # text-only (EPUB → MD, metadata)

# ── PDF converter ─────────────────────────────────────────────────────────────
PDF_DPI            = 300
PDF_MAX_NEW_TOKENS = 4096

PDF_PROMPT = """Convert this PDF page to Markdown. Preserve the original language exactly.

Markdown conventions:
- Headings:  # H1  ## H2  ### H3  #### H4
- Bold: **text**  |  Italic: *text*  |  Bold+italic: ***text***
- Unordered list: - item  |  Ordered list: 1. item
- Table: GFM pipe table with --- separator row
- Inline math: $expr$  |  Block math: $$expr$$
- Image: ![alt](images/filename.ext)
- Footnote: [^n]: text
- Blockquote: > text

Rules:
- Preserve heading hierarchy as in the source
- Replace [IMAGE_N] with ![image_N](images/image_N.png)
- Output ONLY the Markdown, no commentary"""

# ── EPUB converter ────────────────────────────────────────────────────────────
EPUB_MAX_CHUNK_CHARS = 8_000
EPUB_MAX_NEW_TOKENS  = 2_048

EPUB_PROMPT = """Convert the following HTML to Markdown. Preserve the original language exactly.

Markdown conventions:
- Headings:  # H1  ## H2  ### H3  #### H4
- Bold: **text**  |  Italic: *text*  |  Bold+italic: ***text***
- Unordered list: - item  |  Ordered list: 1. item
- Table: GFM pipe table with --- separator row
- Image: ![alt](images/filename.ext) -- place where the <img> appears in the HTML
- Footnote: [^id]: text
- Blockquote: > text
- Code inline: `code`  |  Code block: ```lang\n...\n```

Output only the Markdown.

HTML:
{html}
"""

# ── Evaluation sampling ───────────────────────────────────────────────────────
EVAL_N = 20  # pages/chunks sampled per book for quality evaluation

PDF_JUDGE_PROMPT = (
    "You are a faithfulness verifier for document conversion.\n\n"
    "SOURCE DOCUMENT: the page image above.\n"
    "GENERATED MARKDOWN: the text below.\n\n"
    "Your task: verify that every piece of content in the Markdown can be "
    "traced back to the source page. Do not evaluate markdown quality in general — "
    "only check faithfulness to the original.\n\n"
    "Rate from 1 (unfaithful) to 5 (perfectly faithful):\n"
    "- text: all text from the source is present and nothing is invented\n"
    "- structure: headings, tables, lists match the visual layout of the source\n"
    "- math: formulas match the source (rate 5 if no math is present)\n\n"
    'Return ONLY valid JSON: {"text": N, "structure": N, "math": N}\n\n'
    "GENERATED MARKDOWN:\n"
)

EPUB_JUDGE_PROMPT = (
    "You are a faithfulness verifier for document conversion.\n\n"
    "SOURCE DOCUMENT:\n{html}\n\n"
    "GENERATED MARKDOWN:\n{markdown}\n\n"
    "Your task: verify that every piece of content in the Markdown can be "
    "traced back to the source HTML. Do not evaluate markdown quality in general — "
    "only check faithfulness to the original.\n\n"
    "Rate from 1 (unfaithful) to 5 (perfectly faithful):\n"
    "- text: all text from the source is present and nothing is invented\n"
    "- structure: headings, tables, lists match the source structure\n\n"
    'Return ONLY valid JSON: {"text": N, "structure": N}'
)

# ── Metadata extraction ───────────────────────────────────────────────────────
METADATA_MAX_NEW_TOKENS = 128

# ── Dependency parsing ────────────────────────────────────────────────────────
PARSE_LANGS         = ["it", "de"]
PARSE_OUTPUT_FORMAT = "conllu"  # "conllu", "json", or "both"

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR    = "books/"
OUTPUT_DIR   = "output/"
SCORES_DIR   = "scores/"
METADATA_CSV = "metadata/metadata.csv"