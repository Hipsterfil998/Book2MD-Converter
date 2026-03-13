"""cli.py — Command-line interface for Book2MD Converter.

Usage examples:
    book2md convert --pdf
    book2md convert --epub
    book2md convert --simple
    book2md metadata
    book2md evaluate
    book2md evaluate --bertscore
    book2md parse --langs it de --format conllu
"""

import argparse
from book2md.config import path_config, parse_config


def cmd_convert(args):
    from book2md.pipeline import ConverterPipeline
    pipeline = ConverterPipeline(input_dir=args.input, output_dir=args.output)
    if args.pdf:
        pipeline.run_pdf_llm()
    elif args.epub:
        pipeline.run_epub_llm()
    elif args.simple:
        pipeline.run_simple()


def cmd_metadata(args):
    from book2md.metadata.extractor import MetadataExtractor
    MetadataExtractor().run(output_dir=args.output, output_csv=args.csv)


def cmd_evaluate(args):
    from book2md.evaluation.evaluator import QualityEvaluator
    QualityEvaluator(use_bertscore=args.bertscore).evaluate_all(
        output_dir=args.output, scores_dir=args.scores
    )


def cmd_parse(args):
    from book2md.parsing.parser import DependencyParser
    DependencyParser(langs=args.langs, output_format=args.format).run(
        input_dir=args.output, output_dir=args.parsed
    )


def main():
    parser = argparse.ArgumentParser(
        prog="book2md",
        description="Book2MD Converter — PDF/EPUB to Markdown pipeline",
    )
    parser.add_argument("--input",   default=path_config.input_dir,    metavar="DIR",  help="Input directory (default: %(default)s)")
    parser.add_argument("--output",  default=path_config.output_dir,   metavar="DIR",  help="Output directory (default: %(default)s)")
    parser.add_argument("--scores",  default=path_config.scores_dir,   metavar="DIR",  help="Scores directory (default: %(default)s)")
    parser.add_argument("--csv",     default=path_config.metadata_csv, metavar="FILE", help="Metadata CSV path (default: %(default)s)")
    parser.add_argument("--parsed",  default="parsed/",                metavar="DIR",  help="Parsed output directory (default: %(default)s)")

    sub = parser.add_subparsers(dest="command", required=True)

    # convert
    p_conv = sub.add_parser("convert", help="Convert books to Markdown")
    mode = p_conv.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pdf",    action="store_true", help="PDF → Markdown via LLM (Qwen3-VL)")
    mode.add_argument("--epub",   action="store_true", help="EPUB → Markdown via LLM (Qwen3)")
    mode.add_argument("--simple", action="store_true", help="PDF/EPUB → Markdown rule-based (no LLM)")

    # metadata
    sub.add_parser("metadata", help="Extract author/title/year/genre from converted books")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate conversion quality (NED, BLEU, MarkdownF1)")
    p_eval.add_argument("--bertscore", action="store_true", help="Also compute BERTScore (slower)")

    # parse
    p_parse = sub.add_parser("parse", help="Run dependency parsing on converted Markdown")
    p_parse.add_argument("--langs",  nargs="+", default=parse_config.langs,         metavar="LANG", help="Languages (default: %(default)s)")
    p_parse.add_argument("--format", default=parse_config.output_format, choices=["conllu", "json", "both"], help="Output format (default: %(default)s)")

    args = parser.parse_args()
    {"convert": cmd_convert, "metadata": cmd_metadata, "evaluate": cmd_evaluate, "parse": cmd_parse}[args.command](args)


if __name__ == "__main__":
    main()
