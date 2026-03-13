"""book2md — PDF/EPUB to Markdown conversion pipeline."""

from book2md.pipeline import ConverterPipeline
from book2md.metadata.extractor import MetadataExtractor
from book2md.parsing.parser import DependencyParser
from book2md.evaluation.evaluator import QualityEvaluator

__all__ = ["ConverterPipeline", "MetadataExtractor", "DependencyParser", "QualityEvaluator"]
