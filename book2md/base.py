"""Abstract base class for all pipeline components."""

import gc
from abc import ABC, abstractmethod


class PipelineStep(ABC):
    """Shared interface for converter, metadata, parsing, and evaluation steps."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute the pipeline step."""
        ...

    def cleanup(self) -> None:
        """Release GPU/CPU memory after the step completes."""
        import torch
        gc.collect()
        torch.cuda.empty_cache()
