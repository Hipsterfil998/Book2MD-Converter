# config.py — Central configuration for the book conversion pipeline
# Edit this file to change models, paths, and generation parameters.

# ── Models ───────────────────────────────────────────────────────────────────
PDF_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"  # vision-language (PDF → MD, PDF eval)
TEXT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"     # text-only (EPUB → MD, metadata)

# ── PDF converter ─────────────────────────────────────────────────────────────
PDF_DPI            = 300
PDF_MAX_NEW_TOKENS = 4096

# ── EPUB converter ────────────────────────────────────────────────────────────
EPUB_MAX_CHUNK_CHARS = 8_000
EPUB_MAX_NEW_TOKENS  = 2_048

# ── Evaluation sampling ───────────────────────────────────────────────────────
EVAL_N = 20  # pages/chunks sampled per book for quality evaluation

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
