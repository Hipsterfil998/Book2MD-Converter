"""epub_converter.py — EPUB → Markdown via pypandoc + HuggingFace model."""

from pathlib import Path
import torch
import pypandoc
from bs4 import BeautifulSoup
from ebooklib import epub
from ebooklib.epub import EpubImage
from transformers import AutoProcessor, AutoModelForImageTextToText

PROMPT = """Convert the following HTML to clean Markdown.
Rules:
- Preserve headings, lists, tables and code blocks. 
- when there are images convert them to markdown format ![](images/fig.png)
- put the image flag exactly where it is located in the HTML
- Output only the Markdown.

HTML:
{html}
"""


class EpubToMarkdownConverter:
    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_chunk_chars: int = 8_000,
        max_new_tokens: int = 2_048,
    ):
        self.device = device
        self.max_chunk_chars = max_chunk_chars
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map=device
        )

    def _chunk(self, html: str) -> list[str]:
        """Split HTML into chunks by top-level block tags."""
        soup = BeautifulSoup(html, "html.parser")
        blocks = soup.find_all(["section", "article", "div"], recursive=False)
        if not blocks:
            blocks = soup.find_all(True, recursive=False)

        chunks, current = [], ""
        for block in blocks:
            text = str(block)
            if len(current) + len(text) > self.max_chunk_chars:
                if current:
                    chunks.append(current)
                current = text
            else:
                current += text
        if current:
            chunks.append(current)
        return chunks

    def _infer(self, html_chunk: str) -> str:
        """Run one HTML chunk through the model."""
        messages = [{"role": "user", "content": PROMPT.format(html=html_chunk)}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        ).to(self.device)

        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        new_tokens = out[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def _extract_images(self, epub_path: str, images_dir: Path) -> dict[str, Path]:
        """Extract all images from EPUB to images_dir. Returns {epub_src: local_path}."""
        images_dir.mkdir(parents=True, exist_ok=True)
        book = epub.read_epub(epub_path)
        mapping = {}
        for item in book.get_items():
            if isinstance(item, EpubImage):
                dest = images_dir / Path(item.file_name).name
                dest.write_bytes(item.content)
                mapping[item.file_name] = dest
        return mapping

    def _rewrite_img_srcs(self, html: str, mapping: dict[str, Path]) -> str:
        """Replace img src attributes in HTML to point to extracted local files."""
        soup = BeautifulSoup(html, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src", "")
            # Match by filename since pandoc may strip path prefixes
            match = next((v for k, v in mapping.items() if Path(k).name == Path(src).name), None)
            if match:
                img["src"] = str(match)
        return str(soup)

    def convert(self, epub_path: str, output_path: str | None = None) -> str:
        """Convert EPUB to Markdown. Images are saved alongside output_path."""
        output = Path(output_path) if output_path else Path(epub_path).with_suffix(".md")
        images_dir = output.parent / "images"

        image_map = self._extract_images(epub_path, images_dir)
        html = pypandoc.convert_file(epub_path, "html")
        html = self._rewrite_img_srcs(html, image_map)

        chunks = self._chunk(html)
        markdown = "\n\n".join(self._infer(c) for c in chunks)

        output.write_text(markdown, encoding="utf-8")
        return markdown