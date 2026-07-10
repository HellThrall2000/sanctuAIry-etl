"""Validation configuration and reporting models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationConfig(BaseModel):
    """Configuration model for dataset validation threshold settings.

    Attributes:
        max_message_length: Optional character limit for individual messages.
        max_conversation_length: Optional limit on the number of messages/turns.
    """

    max_message_length: int | None = None
    max_conversation_length: int | None = None


class ConversationValidationResult(BaseModel):
    """Result of validating a single Conversation.

    Attributes:
        conversation_id: The ID of the conversation.
        is_valid: Whether the conversation is valid.
        errors: Detailed validation errors, empty if valid.
    """

    conversation_id: str
    is_valid: bool
    errors: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Structured report returned by a dataset validator containing validation results.

    Attributes:
        num_checked: Total number of conversations checked.
        num_valid: Number of valid conversations.
        num_invalid: Number of invalid conversations.
        results: Dictionary mapping conversation ID to its ConversationValidationResult.
    """

    num_checked: int
    num_valid: int
    num_invalid: int
    results: dict[str, ConversationValidationResult] = Field(default_factory=dict)
