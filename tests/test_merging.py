"""Unit tests for the merging framework."""

from __future__ import annotations

import pytest

from cbt_companion.mergers.base import MergeError
from cbt_companion.mergers.conversation_merger import ConversationMerger
from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message
from cbt_companion.models.merge_report import MergeReport


def test_merger_success() -> None:
    # Arrange: Setup three conversation datasets
    c1 = Conversation(
        id="emp_1",
        source="facebook/empathetic_dialogues",
        metadata=ConversationMetadata(split="train"),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ],
    )
    c2 = Conversation(
        id="emp_2",
        source="facebook/empathetic_dialogues",
        metadata=ConversationMetadata(split="train"),
        messages=[
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="Good"),
        ],
    )
    dataset_1 = [c1, c2]

    c3 = Conversation(
        id="es_1",
        source="thu-coai/esconv",
        metadata=ConversationMetadata(split="validation"),
        messages=[
            Message(role="user", content="Feeling anxious"),
            Message(role="assistant", content="Deep breaths help"),
        ],
    )
    dataset_2 = [c3]

    merger = ConversationMerger()

    # Act
    report = merger.merge([dataset_1, dataset_2])

    # Assert
    assert isinstance(report, MergeReport)
    assert report.total_conversations == 3
    assert len(report.merged_conversations) == 3

    # Order preserved
    assert report.merged_conversations[0].id == "emp_1"
    assert report.merged_conversations[1].id == "emp_2"
    assert report.merged_conversations[2].id == "es_1"

    # Contribution counts
    assert report.contribution_counts == {
        "facebook/empathetic_dialogues": 2,
        "thu-coai/esconv": 1,
    }

    # Dataset order
    assert report.dataset_order == [
        "facebook/empathetic_dialogues",
        "thu-coai/esconv",
    ]


def test_merger_duplicate_ids_raises_error() -> None:
    # Arrange: Two datasets with colliding ID "dup_id"
    c1 = Conversation(
        id="dup_id",
        source="facebook/empathetic_dialogues",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ],
    )
    dataset_1 = [c1]

    c2 = Conversation(
        id="dup_id",
        source="thu-coai/esconv",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Colliding"),
            Message(role="assistant", content="Yes"),
        ],
    )
    dataset_2 = [c2]

    merger = ConversationMerger()

    # Act & Assert
    with pytest.raises(MergeError) as excinfo:
        merger.merge([dataset_1, dataset_2])

    assert "Duplicate conversation ID 'dup_id'" in str(excinfo.value)


def test_merger_handles_empty_dataset() -> None:
    # Arrange: one non-empty dataset and one empty dataset
    c1 = Conversation(
        id="emp_1",
        source="facebook/empathetic_dialogues",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ],
    )
    dataset_1 = [c1]
    dataset_2 = []

    merger = ConversationMerger()

    # Act
    report = merger.merge([dataset_1, dataset_2])

    # Assert
    assert report.total_conversations == 1
    assert report.dataset_order == ["facebook/empathetic_dialogues"]
    assert report.contribution_counts == {"facebook/empathetic_dialogues": 1}
