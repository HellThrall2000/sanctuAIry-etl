"""Validator for canonical conversation structures and thresholds."""

from __future__ import annotations

from cbt_companion.models.conversation import Conversation
from cbt_companion.models.validation_report import ValidationConfig
from cbt_companion.validation.base import BaseValidator


class ConversationValidator(BaseValidator):
    """Structural validator for canonical Conversation objects.

    This validator verifies only structural integrity and makes no assumptions
    about conversation flow, turn sequence, or speaker order.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize the validator.

        Args:
            config: Optional ValidationConfig settings.
        """
        self.config = config or ValidationConfig()

    def validate(self, conversation: Conversation) -> list[str]:
        """Validate a single Conversation for structural integrity.

        Args:
            conversation: The canonical Conversation to validate.

        Returns:
            A list of validation error strings. If empty, the conversation is valid.
        """
        errors = []

        # 1. Null / missing field checks
        if getattr(conversation, "id", None) is None or not str(conversation.id).strip():
            errors.append("Conversation ID is null, empty or missing.")

        if getattr(conversation, "source", None) is None or not str(conversation.source).strip():
            errors.append("Conversation source is null, empty or missing.")

        if getattr(conversation, "metadata", None) is None:
            errors.append("Conversation metadata is null or missing.")

        if getattr(conversation, "messages", None) is None:
            errors.append("Conversation messages list is null or missing.")
            return errors  # Cannot proceed to check messages if list is missing

        # 2. Length check for conversation (number of turns)
        num_messages = len(conversation.messages)
        if num_messages < 2:
            errors.append(f"Conversation has fewer than 2 messages (got {num_messages}).")

        # 4. Message-level checks (roles, empty content)
        for idx, msg in enumerate(conversation.messages):
            if msg is None:
                errors.append(f"Message at index {idx} is null.")
                continue

            role = getattr(msg, "role", None)
            if role is None or role not in ("user", "assistant"):
                errors.append(f"Message at index {idx} has invalid role: '{role}'.")

            content = getattr(msg, "content", None)
            if content is None or not str(content).strip():
                errors.append(f"Message at index {idx} has empty or null content.")
                continue

        return errors
