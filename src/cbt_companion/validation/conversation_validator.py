"""Validator for canonical conversation structures and thresholds."""

from __future__ import annotations

from cbt_companion.models.conversation import Conversation
from cbt_companion.models.validation_report import ValidationConfig
from cbt_companion.validation.base import BaseValidator


class ConversationValidator(BaseValidator):
    """Structural validator for canonical Conversation objects."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize the validator.

        Args:
            config: Optional ValidationConfig threshold settings.
        """
        self.config = config or ValidationConfig()

    def validate(self, conversation: Conversation) -> list[str]:
        """Validate a single Conversation against structural and config thresholds.

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

        if (
            self.config.max_conversation_length is not None
            and num_messages > self.config.max_conversation_length
        ):
            errors.append(
                f"Conversation length ({num_messages}) exceeds maximum limit "
                f"({self.config.max_conversation_length})."
            )

        # 3. Role sequence checks (starts with user, ends with assistant)
        if num_messages >= 1:
            first_role = getattr(conversation.messages[0], "role", None)
            if first_role != "user":
                errors.append(f"Conversation must start with 'user' role (got '{first_role}').")

        if num_messages >= 1:
            last_role = getattr(conversation.messages[-1], "role", None)
            if last_role != "assistant":
                errors.append(f"Conversation must end with 'assistant' role (got '{last_role}').")

        # 4. Message-level checks (roles, empty content, character length)
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

            char_len = len(content)
            if (
                self.config.max_message_length is not None
                and char_len > self.config.max_message_length
            ):
                errors.append(
                    f"Message at index {idx} length ({char_len}) exceeds maximum limit "
                    f"({self.config.max_message_length})."
                )

        return errors
