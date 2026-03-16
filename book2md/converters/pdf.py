"""pdf.py — PDF → Markdown converter using Qwen3-VL via vLLM."""

import logging
from pathlib import Path
import fitz
from pdf2image import convert_from_path
from book2md.config import pdf_config, eval_config
from book2md.base import PipelineStep
from vllm import LLM, SamplingParams
from book2md.utils import md_to_txt, pil_to_data_url, sample_indices, suppress_worker_stderr, truncate_repetitions

logger = logging.getLogger(__name__)


class PDFToMarkdownConverter(PipelineStep):
    """Converts PDF documents to Markdown using Qwen3-VL via vLLM."""

    def __init__(
        self,
        model_id: str = pdf_config.model_id,
        dpi: int = pdf_config.dpi,
        max_new_tokens: int = pdf_config.max_new_tokens,
        eval_n: int = eval_config.n,
    ) -> None:
        self.dpi = dpi
        self.max_new_tokens = max_new_tokens
        self.eval_n = eval_n
        logger.info("Loading %s...", model_id)
        with suppress_worker_stderr():
            self.llm = LLM(model=model_id, dtype="bfloat16", enforce_eager=True,
                           enable_prefix_caching=eval_config.enable_prefix_caching)

    def run(self, pdf_path: str | Path, output_dir: str | Path) -> Path:
        return self.convert(pdf_path, output_dir)

    def convert(self, pdf_path: str | Path, output_dir: str | Path) -> Path:
        """Convert a PDF to Markdown. Returns path to the generated .md file.
        Sampled pages for evaluation are saved to output_dir/eval_pages/."""
        pdf_path, output_dir = Path(pdf_path), Path(output_dir)
        img_dir = output_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        pages = convert_from_path(str(pdf_path), dpi=self.dpi)
        page_images = self._extract_images(pdf_path, img_dir)

        # Filter blank pages before sending to the LLM
        non_blank = [(j, img) for j, img in enumerate(pages) if not self._is_blank(img)]
        skipped = len(pages) - len(non_blank)
        if skipped:
            logger.info("Skipped %d blank page(s)", skipped)

        def make_prompt(fnames: list[str]) -> str:
            if not fnames:
                return pdf_config.prompt
            placeholders = "\n".join(f"[IMAGE_{i + 1}]" for i in range(len(fnames)))
            return pdf_config.prompt + f"\n\nImages in order:\n{placeholders}"

        messages = [
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": pil_to_data_url(img)}},
                {"type": "text", "text": make_prompt(page_images.get(j, []))},
            ]}]
            for j, img in non_blank
        ]

        sampling_params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0,
                                         repetition_penalty=pdf_config.repetition_penalty)
        outputs = self.llm.chat(messages, sampling_params=sampling_params,
                                chat_template_kwargs={"enable_thinking": False})

        raw_texts: dict[int, str] = {}
        markdown_pages = []
        for (j, _), out in zip(non_blank, outputs):
            raw = self._clean(out.outputs[0].text)
            text = self._resolve_image_refs(raw, page_images.get(j, []))
            raw_texts[j] = text
            markdown_pages.append(f"<!-- Page {j + 1} -->\n{text}")

        out_file = output_dir / (pdf_path.stem + ".md")
        md_content = "\n\n".join(markdown_pages)
        out_file.write_text(md_content, encoding="utf-8")
        (output_dir / (pdf_path.stem + ".txt")).write_text(md_to_txt(md_content), encoding="utf-8")
        logger.info("Saved → %s", out_file)

        # Save sampled Markdown pairs for later evaluation.
        # {j}.ref.md: PDF text layer via PyMuPDF Markdown export (no image, no LLM).
        # {j}.md:     LLM-generated Markdown for the same page.
        # Blank pages are excluded from sampling.
        eval_dir = output_dir / "eval_pages"
        eval_dir.mkdir(exist_ok=True)
        non_blank_indices = [j for j, _ in non_blank]
        doc_ref = fitz.open(str(pdf_path))
        for j in sample_indices(len(non_blank_indices), self.eval_n, indices=non_blank_indices):
            (eval_dir / f"{j}.md").write_text(raw_texts[j], encoding="utf-8")
            try:
                ref_md = doc_ref[j].get_text("markdown").strip()
            except (AssertionError, ValueError):
                ref_md = doc_ref[j].get_text("text").strip()  # PyMuPDF < 1.24
            (eval_dir / f"{j}.ref.md").write_text(ref_md, encoding="utf-8")
        doc_ref.close()
        logger.info("Eval pages → %s", eval_dir)

        return out_file

    def _extract_images(self, pdf_path: Path, img_dir: Path) -> dict[int, list[str]]:
        """Extract unique content images from PDF.

        Deduplicates by xref and skips images that appear on > 30% of pages
        (headers, footers, watermarks). Returns page_idx → [filenames].
        """
        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)

        # Count how many distinct pages each xref appears on
        xref_page_count: dict[int, int] = {}
        for page in doc:
            for xref in {img[0] for img in page.get_images(full=True)}:
                xref_page_count[xref] = xref_page_count.get(xref, 0) + 1

        # Exclude recurring decorative elements (headers / footers / watermarks)
        max_pages = max(3, int(n_pages * 0.3))
        content_xrefs = {xref for xref, c in xref_page_count.items() if c <= max_pages}

        # Save each unique content image once
        xref_to_fname: dict[int, str] = {}
        img_counter = 0
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                if xref not in content_xrefs or xref in xref_to_fname:
                    continue
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.colorspace and pix.colorspace.n > 3:  # CMYK or similar → RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    fname = f"img{img_counter + 1}.png"
                    pix.save(str(img_dir / fname))
                except Exception:
                    continue  # skip images with unsupported colorspace
                img_counter += 1
                xref_to_fname[xref] = fname

        page_images: dict[int, list[str]] = {
            page_idx: [
                xref_to_fname[img[0]]
                for img in page.get_images(full=True)
                if img[0] in xref_to_fname
            ]
            for page_idx, page in enumerate(doc)
        }
        doc.close()
        return page_images

    @staticmethod
    def _is_blank(img, white_threshold: int = 240, max_dark_ratio: float = 0.001) -> bool:
        """Return True only if the page is completely white (< 0.1% non-white pixels)."""
        gray = img.convert("L")
        pixels = gray.getdata()
        dark = sum(1 for p in pixels if p < white_threshold)
        return dark / len(pixels) < max_dark_ratio

    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown code fences and truncate repetition loops."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[1:end]).strip()
        return truncate_repetitions(text)

    @staticmethod
    def _resolve_image_refs(text: str, fnames: list[str]) -> str:
        """Replace [IMAGE_N] placeholders; append any unplaced images at the end."""
        for i, fname in enumerate(fnames):
            ref = f"![image_{i + 1}](images/{fname})"
            placeholder = f"[IMAGE_{i + 1}]"
            text = text.replace(placeholder, ref) if placeholder in text else text + f"\n\n{ref}"
        return text
