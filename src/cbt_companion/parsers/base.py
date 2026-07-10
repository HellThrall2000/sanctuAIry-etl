"""Abstract base class for dataset parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cbt_companion.models.raw_conversation import ParseReport


class BaseParser(ABC):
    """Abstract base class for converting raw datasets to raw conversation models."""

    @abstractmethod
    def parse(self, dataset_split: Any, split_name: str) -> ParseReport:
        """Parse a raw dataset split and return a ParseReport.

        Args:
            dataset_split: The raw dataset split (e.g. Hugging Face Dataset).
            split_name: Name of the split being parsed (e.g. "train", "test").

        Returns:
            A ParseReport containing the list of RawConversation objects and statistics.
        """
        pass
