"""Base class and custom exceptions for dataset merging."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from cbt_companion.models.conversation import Conversation
from cbt_companion.models.merge_report import MergeReport


class MergeError(Exception):
    """Raised when an unrecoverable merge collision occurs."""

    pass


class BaseMerger(ABC):
    """Abstract base class for dataset conversation mergers."""

    @abstractmethod
    def merge(self, datasets: Sequence[Sequence[Conversation]]) -> MergeReport:
        """Merge multiple datasets of Conversation objects.

        Args:
            datasets: A sequence of conversation sequences to combine.

        Returns:
            A MergeReport containing the merged conversations and stats.

        Raises:
            MergeError: If duplicate conversation IDs are detected.
        """
        pass
