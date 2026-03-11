"""evaluator.py — Standalone quality evaluator for PDF/EPUB → Markdown conversions.

Reads pre-saved eval_pages/ and eval_chunks/ produced during conversion.
Scores are saved separately from the generated Markdown files.
"""

import json
from pathlib import Path

from PIL import Image
from config import PDF_JUDGE_PROMPT, EPUB_JUDGE_PROMPT, PDF_MODEL_ID, TEXT_MODEL_ID
from vllm import LLM, SamplingParams
from utils import pil_to_data_url, suppress_worker_stderr


class QualityEvaluator:
    """Evaluates PDF/EPUB → Markdown quality using an LLM-as-judge.

    Uses a vision-language model for PDF (image + Markdown pairs) and a
    text-only model for EPUB (HTML + Markdown pairs). Both are loaded lazily
    so that only the required model occupies VRAM.

    Reads pre-saved eval_pages/ (PNG) and eval_chunks/ (HTML) folders
    produced during conversion — no access to original files required.
    """

    def __init__(
        self,
        pdf_judge_model_id: str = PDF_MODEL_ID,
        epub_judge_model_id: str = TEXT_MODEL_ID,
    ):
        self.pdf_judge_model_id = pdf_judge_model_id
        self.epub_judge_model_id = epub_judge_model_id
        self._pdf_llm: LLM | None = None
        self._epub_llm: LLM | None = None

    def _get_pdf_llm(self) -> LLM:
        if self._pdf_llm is None:
            with suppress_worker_stderr():
                self._pdf_llm = LLM(model=self.pdf_judge_model_id, dtype="bfloat16")
        return self._pdf_llm

    def _get_epub_llm(self) -> LLM:
        if self._epub_llm is None:
            with suppress_worker_stderr():
                self._epub_llm = LLM(model=self.epub_judge_model_id, dtype="bfloat16")
        return self._epub_llm

    def evaluate_pdf(self, eval_pages_dir: str | Path, scores_dir: str | Path) -> dict:
        """Evaluate a PDF → Markdown conversion.

        Reads {idx}.png + {idx}.md pairs from eval_pages_dir.
        Returns dict with average and per-page scores.
        """
        eval_pages_dir, scores_dir = Path(eval_pages_dir), Path(scores_dir)
        scores_dir.mkdir(parents=True, exist_ok=True)

        page_files = sorted(eval_pages_dir.glob("*.png"), key=lambda p: int(p.stem))
        messages = [
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": pil_to_data_url(Image.open(pf))}},
                {"type": "text", "text": PDF_JUDGE_PROMPT + pf.with_suffix(".md").read_text(encoding="utf-8")},
            ]}]
            for pf in page_files
        ]
        outputs = self._get_pdf_llm().chat(messages, SamplingParams(max_tokens=64, temperature=0.0))

        fallback = {"text": None, "structure": None, "math": None}
        scores = {}
        for f, out in zip(page_files, outputs):
            try:
                scores[int(f.stem)] = json.loads(out.outputs[0].text.strip())
            except Exception:
                scores[int(f.stem)] = fallback

        result = self._build_result(scores, ("text", "structure", "math"), key="pages")
        (scores_dir / (eval_pages_dir.parent.name + "_scores.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    def evaluate_epub(self, eval_chunks_dir: str | Path, scores_dir: str | Path) -> dict:
        """Evaluate an EPUB → Markdown conversion.

        Reads {idx}.html + {idx}.md pairs from eval_chunks_dir.
        Returns dict with average and per-chunk scores.
        """
        eval_chunks_dir, scores_dir = Path(eval_chunks_dir), Path(scores_dir)
        scores_dir.mkdir(parents=True, exist_ok=True)

        chunk_files = sorted(eval_chunks_dir.glob("*.html"), key=lambda p: int(p.stem))
        messages = [
            [{"role": "user", "content": EPUB_JUDGE_PROMPT.format(
                html=cf.read_text(encoding="utf-8"),
                markdown=cf.with_suffix(".md").read_text(encoding="utf-8"),
            )}]
            for cf in chunk_files
        ]
        outputs = self._get_epub_llm().chat(messages, SamplingParams(max_tokens=64, temperature=0.0))

        fallback = {"text": None, "structure": None}
        scores = {}
        for f, out in zip(chunk_files, outputs):
            try:
                scores[int(f.stem)] = json.loads(out.outputs[0].text.strip())
            except Exception:
                scores[int(f.stem)] = fallback

        result = self._build_result(scores, ("text", "structure"), key="chunks")
        (scores_dir / (eval_chunks_dir.parent.name + "_scores.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    def evaluate_all(self, output_dir: str | Path, scores_dir: str | Path) -> dict:
        """Evaluate all converted books found in output_dir.

        Detects conversion type per book (eval_pages/ = PDF, eval_chunks/ = EPUB).
        Processes all PDFs first, then all EPUBs, so each model is loaded once
        and the two models are not in VRAM simultaneously.

        Returns a dict mapping book name to its scores.
        """
        output_dir, scores_dir = Path(output_dir), Path(scores_dir)
        results = {}

        book_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
        pdf_dirs  = [d for d in book_dirs if (d / "eval_pages").exists()]
        epub_dirs = [d for d in book_dirs if (d / "eval_chunks").exists()]
        skipped   = [d for d in book_dirs if d not in pdf_dirs and d not in epub_dirs]

        for d in skipped:
            print(f"Skipping (no eval data): {d.name}")

        for d in pdf_dirs:
            print(f"Evaluating PDF: {d.name}")
            results[d.name] = self.evaluate_pdf(d / "eval_pages", scores_dir)

        for d in epub_dirs:
            print(f"Evaluating EPUB: {d.name}")
            results[d.name] = self.evaluate_epub(d / "eval_chunks", scores_dir)

        return results

    @staticmethod
    def _build_result(scores: dict, dims: tuple, key: str) -> dict:
        avg = {
            k: round(
                sum(s[k] for s in scores.values() if isinstance(s.get(k), (int, float)))
                / len(scores), 2
            )
            for k in dims
        }
        return {"average": avg, key: {str(j): scores[j] for j in sorted(scores)}}
