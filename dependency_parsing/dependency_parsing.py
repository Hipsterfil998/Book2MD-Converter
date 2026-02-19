#!/usr/bin/env python3
"""
Dependency parsing of txt files using Stanza.
Supports Italian and German. Outputs CoNLL-U and/or JSON.
"""

import argparse
import json
import os
import glob
import stanza


def ensure_model(lang):
    """Download Stanza model if not already present."""
    try:
        stanza.Pipeline(lang=lang, processors="tokenize", download_method=None)
    except Exception:
        print(f"Downloading Stanza model for '{lang}'...")
        stanza.download(lang)


def parse_file(pipeline, filepath):
    """Read a txt file and run the dependency parsing pipeline."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        print(f"  Skipping empty file: {filepath}")
        return None
    doc = pipeline(text)
    return doc


def doc_to_conllu(doc):
    """Convert a Stanza Document to CoNLL-U formatted string."""
    lines = []
    for sentence in doc.sentences:
        for word in sentence.words:
            head = word.head if word.head is not None else 0
            fields = [
                str(word.id),
                word.text,
                word.lemma if word.lemma else "_",
                word.upos if word.upos else "_",
                word.xpos if word.xpos else "_",
                word.feats if word.feats else "_",
                str(head),
                word.deprel if word.deprel else "_",
                "_",  # deps
                "_",  # misc
            ]
            lines.append("\t".join(fields))
        lines.append("")  # blank line between sentences
    return "\n".join(lines)


def doc_to_json(doc):
    """Convert a Stanza Document to a JSON-serialisable structure."""
    sentences = []
    for sentence in doc.sentences:
        tokens = []
        for word in sentence.words:
            tokens.append({
                "id": word.id,
                "text": word.text,
                "lemma": word.lemma,
                "upos": word.upos,
                "xpos": word.xpos,
                "feats": word.feats,
                "head": word.head if word.head is not None else 0,
                "deprel": word.deprel,
            })
        sentences.append({"tokens": tokens})
    return sentences


def main():
    parser = argparse.ArgumentParser(
        description="Run Stanza dependency parsing on a collection of txt files."
    )
    parser.add_argument(
        "-i", "--input-dir",
        default=".",
        help="Directory containing .txt files (default: current directory)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: <input-dir>/parsed_output)",
    )
    parser.add_argument(
        "-l", "--lang",
        nargs="+",
        default=["it", "de"],
        help="Language code(s) for Stanza, e.g. 'it' 'de' (default: it de)",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["conllu", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir or os.path.join(input_dir, "parsed_output")
    os.makedirs(output_dir, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} txt file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Languages: {args.lang}")
    print(f"Format: {args.format}\n")

    # Build a pipeline per language
    pipelines = {}
    for lang in args.lang:
        print(f"Loading Stanza pipeline for '{lang}'...")
        ensure_model(lang)
        pipelines[lang] = stanza.Pipeline(
            lang=lang,
            processors="tokenize,mwt,pos,lemma,depparse",
            download_method=None,
        )

    for filepath in txt_files:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        print(f"Processing: {os.path.basename(filepath)}")

        for lang in args.lang:
            doc = parse_file(pipelines[lang], filepath)
            if doc is None:
                continue

            suffix = f"_{lang}" if len(args.lang) > 1 else ""

            if args.format in ("conllu", "both"):
                conllu_path = os.path.join(output_dir, f"{basename}{suffix}.conllu")
                with open(conllu_path, "w", encoding="utf-8") as f:
                    f.write(doc_to_conllu(doc))
                print(f"  -> {conllu_path}")

            if args.format in ("json", "both"):
                json_path = os.path.join(output_dir, f"{basename}{suffix}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"file": os.path.basename(filepath), "lang": lang,
                         "sentences": doc_to_json(doc)},
                        f, ensure_ascii=False, indent=2,
                    )
                print(f"  -> {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
