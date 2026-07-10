"""Unit tests for the validation framework."""

from __future__ import annotations

import pytest

from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message
from cbt_companion.models.validation_report import ValidationConfig, ValidationReport
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


def test_validator_structural_failures() -> None:
    # Arrange: Less than 2 messages
    c_short = Conversation(
        id="short",
        source="src",
        metadata=ConversationMetadata(),
        messages=[Message(role="user", content="Hello")],
    )

    # Arrange: Starts with assistant
    c_start_assistant = Conversation(
        id="start_assistant",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="assistant", content="Hello"),
            Message(role="assistant", content="World"),
        ],
    )

    # Arrange: Ends with user
    c_end_user = Conversation(
        id="end_user",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="user", content="World"),
        ],
    )

    validator = ConversationValidator()

    # Act & Assert
    err_short = validator.validate(c_short)
    assert any("fewer than 2 messages" in e for e in err_short)
    # also flags not ending with assistant because there's only 1 message which has role user
    assert any("end with 'assistant' role" in e for e in err_short)

    err_start = validator.validate(c_start_assistant)
    assert any("start with 'user' role" in e for e in err_start)

    err_end = validator.validate(c_end_user)
    assert any("end with 'assistant' role" in e for e in err_end)


def test_validator_null_and_empty_fields() -> None:
    # Arrange: Create conversation with empty fields bypassing pydantic check via model_construct
    c_empty_fields = Conversation.model_construct(
        id="  ",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="World"),
        ],
    )

    validator = ConversationValidator()

    # Act
    errors = validator.validate(c_empty_fields)

    # Assert
    assert any("Conversation ID is null, empty or missing" in e for e in errors)


def test_validator_configurable_lengths() -> None:
    # Arrange: Conversation with 4 messages, ending with assistant
    messages = [
        Message(role="user", content="Hello there!"),
        Message(role="assistant", content="How can I assist you today?"),
        Message(role="user", content="I'm feeling down."),
        Message(role="assistant", content="I am sorry to hear that."),
    ]
    conv = Conversation(
        id="conv_len",
        source="src",
        metadata=ConversationMetadata(),
        messages=messages,
    )

    # Config 1: max_conversation_length = 2
    config_conv_len = ValidationConfig(max_conversation_length=2)
    validator_conv = ConversationValidator(config=config_conv_len)

    # Config 2: max_message_length = 10
    config_msg_len = ValidationConfig(max_message_length=10)
    validator_msg = ConversationValidator(config=config_msg_len)

    # Act & Assert
    errors_conv = validator_conv.validate(conv)
    assert any("Conversation length (4) exceeds maximum limit (2)" in e for e in errors_conv)

    errors_msg = validator_msg.validate(conv)
    # All 4 messages exceed 10 chars
    assert len(errors_msg) == 4
    assert any("Message at index 0 length (12) exceeds maximum limit (10)" in e for e in errors_msg)


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
            Message(role="assistant", content="World"),
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
