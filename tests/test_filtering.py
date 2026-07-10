"""Unit tests for the filtering framework."""

from __future__ import annotations

import pytest

from cbt_companion.filters.base import FilterError
from cbt_companion.filters.conversation_filter import ConversationFilter
from cbt_companion.filters.registry import FilterRegistry
from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message
from cbt_companion.models.filter_report import FilterReport
from cbt_companion.models.validation_report import ConversationValidationResult, ValidationReport


def test_filtering_success() -> None:
    # Arrange: 3 conversations
    c1 = Conversation(
        id="c1",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="World"),
        ],
    )
    c2 = Conversation(
        id="c2",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Test"),
            Message(role="assistant", content="Response"),
        ],
    )
    c3 = Conversation(
        id="c3",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Another"),
            Message(role="assistant", content="Turn"),
        ],
    )

    # validation report maps c1: valid, c2: invalid, c3: valid
    val_report = ValidationReport(
        num_checked=3,
        num_valid=2,
        num_invalid=1,
        results={
            "c1": ConversationValidationResult(conversation_id="c1", is_valid=True),
            "c2": ConversationValidationResult(
                conversation_id="c2", is_valid=False, errors=["Error"]
            ),
            "c3": ConversationValidationResult(conversation_id="c3", is_valid=True),
        },
    )

    filt = ConversationFilter()

    # Act
    report = filt.filter([c1, c2, c3], val_report)

    # Assert
    assert isinstance(report, FilterReport)
    assert report.num_kept == 2
    assert report.num_removed == 1
    assert report.removed_ids == ["c2"]
    assert len(report.filtered_conversations) == 2

    # Original order is preserved (c1, c3)
    assert report.filtered_conversations[0].id == "c1"
    assert report.filtered_conversations[1].id == "c3"


def test_filtering_missing_result_raises_error() -> None:
    # Arrange: 2 conversations
    c1 = Conversation(
        id="c1",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="World"),
        ],
    )
    c2 = Conversation(
        id="c2",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Test"),
            Message(role="assistant", content="Response"),
        ],
    )

    # val_report only contains result for c1, missing c2
    val_report = ValidationReport(
        num_checked=1,
        num_valid=1,
        num_invalid=0,
        results={
            "c1": ConversationValidationResult(conversation_id="c1", is_valid=True),
        },
    )

    filt = ConversationFilter()

    # Act & Assert
    with pytest.raises(FilterError) as excinfo:
        filt.filter([c1, c2], val_report)

    assert "missing from the validation report" in str(excinfo.value)
    assert "c2" in str(excinfo.value)


def test_filter_registry() -> None:
    # Act & Assert
    assert FilterRegistry.get_filter_class("conversation") is ConversationFilter
    assert FilterRegistry.get_filter_class("CONVERSATION") is ConversationFilter

    with pytest.raises(KeyError):
        FilterRegistry.get_filter_class("nonexistent_filter")
