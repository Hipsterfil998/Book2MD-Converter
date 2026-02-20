"""
Dependency parsing of Markdown files using Stanza.
Markdown is converted to plain text before parsing.
Supports Italian and German. Outputs CoNLL-U and/or JSON.
"""

import json
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
import stanza


class DependencyParser:

    def __init__(self, langs: list = ["it", "de"], output_format: str = "both"):
        """
        Args:
            langs: list of Stanza language codes, e.g. ["it", "de"]
            output_format: "conllu", "json", or "both"
        """
        self.langs = langs
        self.output_format = output_format
        self.pipelines = {}

    def _md_to_txt(self, md_text: str) -> str:
        """Convert Markdown to plain text."""
        html = markdown.markdown(md_text)
        return BeautifulSoup(html, "html.parser").get_text(separator="\n")

    def _ensure_model(self, lang: str) -> None:
        """Download Stanza model if not already present."""
        try:
            stanza.Pipeline(lang=lang, processors="tokenize", download_method=None)
        except Exception:
            print(f"Downloading Stanza model for '{lang}'...")
            stanza.download(lang)

    def _load_pipelines(self) -> None:
        """Initialize one Stanza pipeline per language."""
        for lang in self.langs:
            print(f"Loading Stanza pipeline for '{lang}'...")
            self._ensure_model(lang)
            self.pipelines[lang] = stanza.Pipeline(
                lang=lang,
                processors="tokenize,mwt,pos,lemma,depparse",
                download_method=None,
            )

    def _doc_to_conllu(self, doc) -> str:
        """Convert a Stanza Document to CoNLL-U string."""
        lines = []
        for sentence in doc.sentences:
            for word in sentence.words:
                fields = [
                    str(word.id),
                    word.text,
                    word.lemma or "_",
                    word.upos or "_",
                    word.xpos or "_",
                    word.feats or "_",
                    str(word.head if word.head is not None else 0),
                    word.deprel or "_",
                    "_",
                    "_",
                ]
                lines.append("\t".join(fields))
            lines.append("")
        return "\n".join(lines)

    def _doc_to_json(self, doc, source_file: str, lang: str) -> dict:
        """Convert a Stanza Document to a JSON-serialisable dict."""
        sentences = []
        for sentence in doc.sentences:
            tokens = [
                {
                    "id": word.id,
                    "text": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats,
                    "head": word.head if word.head is not None else 0,
                    "deprel": word.deprel,
                }
                for word in sentence.words
            ]
            sentences.append({"tokens": tokens})
        return {"file": source_file, "lang": lang, "sentences": sentences}

    def run(self, input_dir: str, output_dir: str = None) -> None:
        """
        Parse all .md files in input_dir.
        For each file: convert md -> txt, then run dependency parsing.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else input_path / "parsed_output"
        output_path.mkdir(parents=True, exist_ok=True)

        md_files = sorted(input_path.glob("**/*.md"))
        if not md_files:
            print(f"No .md files found in {input_dir}")
            return

        print(f"Found {len(md_files)} markdown file(s)")
        print(f"Output: {output_path}\n")

        self._load_pipelines()

        for md_file in md_files:
            print(f"Processing: {md_file.name}")
            md_text = md_file.read_text(encoding="utf-8")
            text = self._md_to_txt(md_text)

            if not text.strip():
                print(f"  Skipping empty file: {md_file.name}")
                continue

            stem = md_file.stem
            for lang in self.langs:
                doc = self.pipelines[lang](text)
                suffix = f"_{lang}" if len(self.langs) > 1 else ""

                if self.output_format in ("conllu", "both"):
                    out = output_path / f"{stem}{suffix}.conllu"
                    out.write_text(self._doc_to_conllu(doc), encoding="utf-8")
                    print(f"  -> {out}")

                if self.output_format in ("json", "both"):
                    out = output_path / f"{stem}{suffix}.json"
                    out.write_text(
                        json.dumps(self._doc_to_json(doc, md_file.name, lang),
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"  -> {out}")

        print("\nDone.")
