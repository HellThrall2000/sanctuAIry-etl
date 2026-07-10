"""Abstract base class for dataset validators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cbt_companion.models.conversation import Conversation
from cbt_companion.models.validation_report import ConversationValidationResult, ValidationReport


class BaseValidator(ABC):
    """Abstract base class for Conversation validation."""

    @abstractmethod
    def validate(self, conversation: Conversation) -> list[str]:
        """Validate a single canonical Conversation.

        Args:
            conversation: The conversation to validate.

        Returns:
            A list of validation error strings. If empty, the conversation is valid.
        """
        pass

    def validate_dataset(self, conversations: list[Conversation]) -> ValidationReport:
        """Validate a list of conversations and compile a report.

        This method does NOT filter or modify the list of conversations.
        It returns results for every conversation.

        Args:
            conversations: The list of conversations to check.

        Returns:
            A ValidationReport containing checked, valid, and invalid counts and results.
        """
        results = {}
        num_valid = 0
        num_invalid = 0

        for conv in conversations:
            errors = self.validate(conv)
            is_valid = len(errors) == 0
            if is_valid:
                num_valid += 1
            else:
                num_invalid += 1

            results[conv.id] = ConversationValidationResult(
                conversation_id=conv.id,
                is_valid=is_valid,
                errors=errors,
            )

        return ValidationReport(
            num_checked=len(conversations),
            num_valid=num_valid,
            num_invalid=num_invalid,
            results=results,
        )
