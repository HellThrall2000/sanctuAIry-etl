"""Base class and custom exceptions for dataset filtering."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cbt_companion.models.conversation import Conversation
from cbt_companion.models.filter_report import FilterReport
from cbt_companion.models.validation_report import ValidationReport


class FilterError(Exception):
    """Raised when there is an unrecoverable filtering/pipeline inconsistency."""

    pass


class BaseFilter(ABC):
    """Abstract base class for dataset conversation filtering."""

    @abstractmethod
    def filter(
        self,
        conversations: list[Conversation],
        validation_report: ValidationReport,
    ) -> FilterReport:
        """Filter the list of conversations using the validation report.

        Args:
            conversations: Input list of canonical Conversation objects.
            validation_report: The ValidationReport containing checks results.

        Returns:
            A FilterReport containing retained conversations and statistics.

        Raises:
            FilterError: If a conversation's validation result is missing.
        """
        pass
