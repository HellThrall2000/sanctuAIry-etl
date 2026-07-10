"""Filter implementation based on ValidationReport results."""

from __future__ import annotations

from cbt_companion.filters.base import BaseFilter, FilterError
from cbt_companion.models.conversation import Conversation
from cbt_companion.models.filter_report import FilterReport
from cbt_companion.models.validation_report import ValidationReport


class ConversationFilter(BaseFilter):
    """Filters conversations by removing those marked invalid in a ValidationReport."""

    def filter(
        self,
        conversations: list[Conversation],
        validation_report: ValidationReport,
    ) -> FilterReport:
        """Filter out conversations that failed validation.

        Args:
            conversations: Input list of canonical Conversation objects.
            validation_report: Report containing validation results.

        Returns:
            A FilterReport with the filtered conversations and statistics.

        Raises:
            FilterError: If any conversation is missing from the validation report.
        """
        filtered_conversations = []
        removed_ids = []

        for conv in conversations:
            if conv.id not in validation_report.results:
                raise FilterError(
                    f"Pipeline inconsistency: Conversation ID '{conv.id}' "
                    f"is missing from the validation report."
                )

            result = validation_report.results[conv.id]
            if not result.is_valid:
                removed_ids.append(conv.id)
            else:
                filtered_conversations.append(conv)

        return FilterReport(
            num_kept=len(filtered_conversations),
            num_removed=len(removed_ids),
            removed_ids=removed_ids,
            filtered_conversations=filtered_conversations,
        )
