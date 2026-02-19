"""
pdf_to_markdown.py — PDF → Markdown converter using Gemma 3 VL.
"""

import logging
from pathlib import Path
import fitz
import torch
from pdf2image import convert_from_path
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


class PDFToMarkdownConverter:
    """Converts PDF documents to Markdown using Gemma 3 VL."""

    _PROMPT = """Convert this PDF page to Markdown.
                Rules:
                - Preserve heading hierarchy (# ## ###)
                - Convert tables to Markdown tables
                - Use $...$ for inline math, $$...$$ for block math
                - Preserve lists and indentation
                - Replace [IMAGE_N] placeholders with ![image_N](images/image_N.png)
                - Output ONLY the Markdown, no commentary"""

    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        device: str = "cuda",
        dpi: int = 300,
        max_new_tokens: int = 4096,
        batch_size: int = 16,
    ) -> None:
        self.dpi = dpi
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        logger.info("Loading %s...", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device
        ).eval()

    def convert(self, pdf_path: str | Path, output_dir: str | Path) -> Path:
        """Convert a PDF to Markdown. Returns path to the generated .md file."""
        pdf_path, output_dir = Path(pdf_path), Path(output_dir)
        img_dir = output_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        pages = convert_from_path(str(pdf_path), dpi=self.dpi)
        page_images = self._extract_images(pdf_path, img_dir)
        markdown_pages = []

        for i in range(0, len(pages), self.batch_size):
            batch = pages[i : i + self.batch_size]
            logger.info("Pages %d–%d / %d", i + 1, i + len(batch), len(pages))
            results = self._run_batch(batch, i, page_images)
            for j, text in enumerate(results):
                text = self._resolve_image_refs(text, page_images.get(i + j, []))
                markdown_pages.append(f"<!-- Page {i + j + 1} -->\n{text}")

        out_file = output_dir / (pdf_path.stem + ".md")
        out_file.write_text("\n\n---\n\n".join(markdown_pages), encoding="utf-8")
        logger.info("Saved → %s", out_file)
        return out_file


    def _extract_images(self, pdf_path: Path, img_dir: Path) -> dict[int, list[str]]:
        """Extract embedded images from PDF. Returns page_idx → [filenames]."""
        page_images: dict[int, list[str]] = {}
        doc = fitz.open(str(pdf_path))
        for page_idx, page in enumerate(doc):
            page_images[page_idx] = []
            for img_idx, img in enumerate(page.get_images(full=True)):
                pix = fitz.Pixmap(doc, img[0])
                if pix.n > 4:  # CMYK → RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                fname = f"page{page_idx + 1}_img{img_idx + 1}.png"
                pix.save(str(img_dir / fname))
                page_images[page_idx].append(fname)
        doc.close()
        return page_images

    def _run_batch(self, pages: list, offset: int, page_images: dict) -> list[str]:
        """Run batched inference; return decoded text per page."""
        def make_prompt(fnames: list[str]) -> str:
            if not fnames:
                return self._PROMPT
            placeholders = "\n".join(f"[IMAGE_{i+1}]" for i in range(len(fnames)))
            return self._PROMPT + f"\n\nImages in order:\n{placeholders}"

        messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": make_prompt(page_images.get(offset + j, []))},
            ]}]
            for j, img in enumerate(pages)
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, padding=True, truncation=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        input_len = inputs["input_ids"].shape[-1]
        return [self.processor.decode(output_ids[j, input_len:], skip_special_tokens=True) for j in range(len(pages))]

    @staticmethod
    def _resolve_image_refs(text: str, fnames: list[str]) -> str:
        """Replace [IMAGE_N] placeholders; append any unplaced images at the end."""
        for i, fname in enumerate(fnames):
            ref = f"![image_{i+1}](images/{fname})"
            placeholder = f"[IMAGE_{i+1}]"
            text = text.replace(placeholder, ref) if placeholder in text else text + f"\n\n{ref}"
        return text