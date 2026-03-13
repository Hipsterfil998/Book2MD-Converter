"""
conftest.py — Pytest configuration for the book2md pipeline.

Mocks heavy/GPU dependencies (vllm, fitz, stanza, metrics, …) before any module
imports them, so tests can run on a plain CPU machine without model weights.
"""
import sys
from unittest.mock import MagicMock

# ── Mock heavy dependencies ────────────────────────────────────────────────────
for _mod in ("vllm", "fitz", "pdf2image", "pypandoc", "stanza", "langdetect"):
    sys.modules[_mod] = MagicMock()

# langdetect: expose the specific names imported by parser.py
_langdetect = sys.modules["langdetect"]
_langdetect.detect = MagicMock(return_value="it")
_langdetect.LangDetectException = Exception
_langdetect.DetectorFactory = MagicMock()

# Page2MDBench metrics (not available in test environment)
sys.modules["metrics"] = MagicMock()

# ebooklib: give EpubImage / EpubHtml real types so isinstance() works
_epub = MagicMock()
_epub.EpubImage = type("EpubImage", (), {})
_epub.EpubHtml  = type("EpubHtml",  (), {})
sys.modules["ebooklib"]      = MagicMock()
sys.modules["ebooklib.epub"] = _epub
