"""Unit tests for the CanonicalDatasetExporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbt_companion.analytics.canonical_exporter import CanonicalDatasetExporter
from cbt_companion.models.conversation import (
    Conversation,
    ConversationMetadata,
    Message,
)


@pytest.fixture
def sample_conversations() -> list[Conversation]:
    return [
        Conversation(
            id="conv_1",
            source="dataset_a",
            metadata=ConversationMetadata(
                split="train",
                source_dataset="test/dataset_a",
                original_id="orig_1",
                context="greeting",
            ),
            messages=[
                Message(role="user", content="Hello!"),
                Message(role="assistant", content="Hi there!"),
            ],
        ),
        Conversation(
            id="conv_2",
            source="dataset_b",
            metadata=ConversationMetadata(
                split="train",
                source_dataset="test/dataset_b",
                original_id="orig_2",
                context="farewell",
            ),
            messages=[
                Message(role="user", content="Goodbye."),
                Message(role="assistant", content="See you!"),
                Message(role="user", content="Take care."),
            ],
        ),
        Conversation(
            id="conv_3",
            source="dataset_a",
            metadata=ConversationMetadata(
                split="validation",
                source_dataset="test/dataset_a",
            ),
            messages=[
                Message(role="assistant", content="How can I help?"),
                Message(role="user", content="I need advice."),
            ],
        ),
    ]


def test_export_produces_valid_jsonl(
    tmp_path: Path, sample_conversations: list[Conversation]
) -> None:
    """Verify each line is valid JSON and the file contains the correct number of lines."""
    output_file = tmp_path / "output" / "canonical_dataset.jsonl"
    count = CanonicalDatasetExporter.export(sample_conversations, output_file)

    assert count == 3
    assert output_file.exists()

    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)


def test_export_correct_conversation_count(
    tmp_path: Path, sample_conversations: list[Conversation]
) -> None:
    """Verify the return value and number of JSONL lines match the input count."""
    output_file = tmp_path / "test.jsonl"
    count = CanonicalDatasetExporter.export(sample_conversations, output_file)

    assert count == len(sample_conversations)

    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(sample_conversations)


def test_export_one_conversation_per_line(
    tmp_path: Path, sample_conversations: list[Conversation]
) -> None:
    """Verify each line deserialises to a single conversation with a unique ID."""
    output_file = tmp_path / "test.jsonl"
    CanonicalDatasetExporter.export(sample_conversations, output_file)

    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == ["conv_1", "conv_2", "conv_3"]


def test_export_preserves_metadata(
    tmp_path: Path, sample_conversations: list[Conversation]
) -> None:
    """Verify all metadata fields survive the serialisation round-trip."""
    output_file = tmp_path / "test.jsonl"
    CanonicalDatasetExporter.export(sample_conversations, output_file)

    lines = output_file.read_text(encoding="utf-8").strip().split("\n")

    # Check first conversation metadata
    first = json.loads(lines[0])
    assert first["id"] == "conv_1"
    assert first["source"] == "dataset_a"
    assert first["metadata"]["split"] == "train"
    assert first["metadata"]["source_dataset"] == "test/dataset_a"
    assert first["metadata"]["original_id"] == "orig_1"
    assert first["metadata"]["context"] == "greeting"
    assert len(first["messages"]) == 2
    assert first["messages"][0]["role"] == "user"
    assert first["messages"][0]["content"] == "Hello!"

    # Check second conversation has 3 messages
    second = json.loads(lines[1])
    assert len(second["messages"]) == 3

    # Check third conversation with null optional metadata
    third = json.loads(lines[2])
    assert third["metadata"]["context"] is None
    assert third["metadata"]["original_id"] is None

    # Verify full round-trip: JSON -> Conversation -> compare
    for line, original in zip(lines, sample_conversations):
        restored = Conversation.model_validate_json(line)
        assert restored == original


def test_export_creates_output_directory(
    tmp_path: Path, sample_conversations: list[Conversation]
) -> None:
    """Verify the exporter creates missing parent directories automatically."""
    nested_path = tmp_path / "a" / "b" / "c" / "output.jsonl"
    assert not nested_path.parent.exists()

    CanonicalDatasetExporter.export(sample_conversations, nested_path)

    assert nested_path.parent.exists()
    assert nested_path.exists()


def test_export_empty_dataset(tmp_path: Path) -> None:
    """Verify exporting an empty dataset produces an empty file."""
    output_file = tmp_path / "empty.jsonl"
    count = CanonicalDatasetExporter.export([], output_file)

    assert count == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == ""
