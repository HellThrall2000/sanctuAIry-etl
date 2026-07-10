"""Unit tests for the validation framework."""

from __future__ import annotations

import pytest

from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message
from cbt_companion.models.validation_report import ValidationReport
from cbt_companion.validation.conversation_validator import ConversationValidator
from cbt_companion.validation.registry import ValidatorRegistry


def test_validator_success() -> None:
    # Arrange: Valid conversation
    messages = [
        Message(role="user", content="Hi, I need help."),
        Message(role="assistant", content="Of course, I am here to listen."),
    ]
    conv = Conversation(
        id="conv_valid",
        source="dataset_abc",
        metadata=ConversationMetadata(split="train"),
        messages=messages,
    )

    validator = ConversationValidator()

    # Act
    errors = validator.validate(conv)

    # Assert
    assert len(errors) == 0


def test_validator_accepts_any_conversation_flow() -> None:
    # Arrange: First message is assistant
    c_start_assistant = Conversation(
        id="start_assistant",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="assistant", content="Hello"),
            Message(role="assistant", content="World"),
        ],
    )

    # Arrange: Last message is user
    c_end_user = Conversation(
        id="end_user",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="user", content="World"),
        ],
    )

    # Arrange: First is assistant and last is user
    c_both = Conversation(
        id="both",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="assistant", content="How are you?"),
            Message(role="user", content="Good."),
        ],
    )

    validator = ConversationValidator()

    # Act & Assert
    assert len(validator.validate(c_start_assistant)) == 0
    assert len(validator.validate(c_end_user)) == 0
    assert len(validator.validate(c_both)) == 0


def test_validator_structural_failures() -> None:
    # Arrange: fewer than 2 messages
    c_short = Conversation(
        id="short",
        source="src",
        metadata=ConversationMetadata(),
        messages=[Message(role="user", content="Hello")],
    )

    # Arrange: invalid role
    c_invalid_role = Conversation(
        id="invalid_role",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message.model_construct(role="invalid_role", content="World"),
        ],
    )

    # Arrange: empty/whitespace-only content
    c_empty_content = Conversation(
        id="empty_content",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message.model_construct(role="assistant", content="   "),
        ],
    )

    # Arrange: missing required fields (ID)
    c_missing_id = Conversation.model_construct(
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="World"),
        ],
    )

    # Arrange: null required fields (metadata, messages)
    c_null_fields = Conversation.model_construct(
        id="null_fields",
        source="src",
        metadata=None,
        messages=None,
    )

    validator = ConversationValidator()

    # Act & Assert
    err_short = validator.validate(c_short)
    assert any("fewer than 2 messages" in e for e in err_short)

    err_role = validator.validate(c_invalid_role)
    assert any("invalid role" in e for e in err_role)

    err_empty = validator.validate(c_empty_content)
    assert any("empty or null content" in e for e in err_empty)

    err_missing = validator.validate(c_missing_id)
    assert any("Conversation ID is null, empty or missing" in e for e in err_missing)

    err_null = validator.validate(c_null_fields)
    assert any("Conversation metadata is null or missing" in e for e in err_null)
    assert any("Conversation messages list is null or missing" in e for e in err_null)



def test_validate_dataset_report() -> None:
    # Arrange: 1 valid, 1 invalid conversation
    c_valid = Conversation(
        id="valid",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="World"),
        ],
    )
    c_invalid = Conversation(
        id="invalid",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
        ],
    )

    validator = ConversationValidator()

    # Act
    report = validator.validate_dataset([c_valid, c_invalid])

    # Assert
    assert isinstance(report, ValidationReport)
    assert report.num_checked == 2
    assert report.num_valid == 1
    assert report.num_invalid == 1
    assert "valid" in report.results
    assert report.results["valid"].is_valid is True
    assert len(report.results["valid"].errors) == 0

    assert "invalid" in report.results
    assert report.results["invalid"].is_valid is False
    assert len(report.results["invalid"].errors) > 0


def test_validator_registry() -> None:
    # Act & Assert
    assert ValidatorRegistry.get_validator_class("conversation") is ConversationValidator
    assert ValidatorRegistry.get_validator_class("CONVERSATION") is ConversationValidator

    with pytest.raises(KeyError):
        ValidatorRegistry.get_validator_class("nonexistent_validator")
