"""
conftest.py — Pytest configuration for the book-conversion pipeline.

Mocks heavy/GPU dependencies (vllm, fitz, stanza, …) before any module
imports them, so tests can run on a plain CPU machine without model weights.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make bare project imports (from config import ..., from utils import ...) work
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Mock heavy dependencies ────────────────────────────────────────────────────
for _mod in ("vllm", "fitz", "pdf2image", "pypandoc", "stanza"):
    sys.modules[_mod] = MagicMock()

# ebooklib: give EpubImage / EpubHtml real types so isinstance() works
_epub = MagicMock()
_epub.EpubImage = type("EpubImage", (), {})
_epub.EpubHtml  = type("EpubHtml",  (), {})
sys.modules["ebooklib"]      = MagicMock()
sys.modules["ebooklib.epub"] = _epub
