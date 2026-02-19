import argparse
import fitz
from markdownify import markdownify
from pathlib import Path
import glob
from ebooklib import epub
from ebooklib.epub import EpubHtml



class DocumentProcessor:
    """Process PDF and EPUB documents to Markdown"""

    @staticmethod
    def _validate_output(input_count, output_count, content_type="files"):
        """Validate and report processing completion"""
        if input_count == output_count:
            print(f"All the {output_count} {content_type} have been processed")
        else:
            missing = list(range(output_count, input_count))
            print(f"{input_count - output_count} are missing. Missing indices: {missing}")

    def process_pdf(self, pdf_path, output_dir='output', min_size=50):
        """Extract markdown and images from PDF"""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        filename = Path(pdf_path).stem
        doc = fitz.open(pdf_path)
        pages = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text("markdown")
            images = []

            for img_idx, img_info in enumerate(page.get_images(full=True)):
                try:
                    img = doc.extract_image(img_info[0])
                    if img['width'] < min_size or img['height'] < min_size:
                        continue
                    img_dir = output / 'images'
                    img_dir.mkdir(exist_ok=True)
                    img_filename = f"p{page_num:04d}_i{img_idx + 1}.{img['ext']}"
                    (img_dir / img_filename).write_bytes(img['image'])
                    images.append(img_filename)
                except:
                    pass

            pages.append({'page': page_num, 'text': text, 'images': images})

        doc.close()

        with open(output / f'{filename}.md', 'w', encoding='utf-8') as f:
            for p in pages:
                f.write(p['text'])
                if p['images']:
                    f.write('\n' + '\n'.join(f"![](images/{img})" for img in p['images']) + '\n')

        return pages

    def process_epub(self, epub_path, output_dir='output'):
        """Extract markdown from EPUB"""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        filename = Path(epub_path).stem
        book = epub.read_epub(epub_path)
        chapters = []

        for item in book.get_items():
            if isinstance(item, EpubHtml):
                html = item.get_content().decode('utf-8', errors='ignore')
                md = markdownify(html, heading_style="ATX", strip=['script', 'style', 'nav', 'footer', 'header'])
                if md.strip():
                    chapters.append(md)

        with open(output / f'{filename}.md', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(chapters))

        return chapters
    
