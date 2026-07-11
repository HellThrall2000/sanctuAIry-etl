"""Unit tests for the exact conversation deduplicator."""

from __future__ import annotations

from cbt_companion.analytics.deduplicator import ConversationDeduplicator
from cbt_companion.models.conversation import (
    Conversation,
    ConversationMetadata,
    Message,
)


def test_normalize_content_whitespace_and_lowercase() -> None:
    # Arrange & Act & Assert
    # 1. Whitespace collapsing and trimming
    assert (
        ConversationDeduplicator.normalize_content("   hello   world   ") == "hello world"
    )
    assert ConversationDeduplicator.normalize_content("\t hello \n world \t") == "hello world"

    # 2. Lowercase normalization
    assert ConversationDeduplicator.normalize_content("Hello WORLD") == "hello world"

    # 3. Line ending normalization
    assert (
        ConversationDeduplicator.normalize_content("hello\r\nworld\r\neveryone")
        == "hello world everyone"
    )


def test_compute_hash_ignores_metadata_and_is_deterministic() -> None:
    # Arrange
    conv1 = Conversation(
        id="c1",
        source="source_a",
        metadata=ConversationMetadata(
            split="train",
            source_dataset="dataset_1",
            original_id="orig_1",
            context="fear",
        ),
        messages=[
            Message(role="user", content="Hello World"),
            Message(role="assistant", content="How are you?"),
        ],
    )

    conv2 = Conversation(
        id="c2",  # Different ID
        source="source_b",  # Different Source
        metadata=ConversationMetadata(
            split="validation",  # Different Split
            source_dataset="dataset_2",  # Different Dataset
            original_id="orig_2",  # Different original_id
            context="joy",  # Different Context
        ),
        messages=[
            Message(role="user", content="hello   world"),  # Extra whitespace/case diff
            Message(role="assistant", content="HOW ARE YOU?\r\n"),
        ],
    )

    # Act
    hash1 = ConversationDeduplicator.compute_hash(conv1.messages)
    hash2 = ConversationDeduplicator.compute_hash(conv2.messages)

    # Assert
    assert hash1 == hash2
    # Deterministic check: Hash is reproducible
    assert hash1 == ConversationDeduplicator.compute_hash(conv1.messages)


def test_different_punctuation_or_wording_produces_different_hashes() -> None:
    # Arrange
    conv1 = Conversation(
        id="c1",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello world!"),
        ],
    )
    conv2 = Conversation(
        id="c2",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello world"),
        ],
    )
    conv3 = Conversation(
        id="c3",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="Hello there world"),
        ],
    )

    # Act & Assert
    h1 = ConversationDeduplicator.compute_hash(conv1.messages)
    h2 = ConversationDeduplicator.compute_hash(conv2.messages)
    h3 = ConversationDeduplicator.compute_hash(conv3.messages)

    assert h1 != h2
    assert h2 != h3
    assert h1 != h3


def test_deduplicate_keeps_first_occurrence_and_preserves_order() -> None:
    # Arrange
    conv1 = Conversation(
        id="first_unique",
        source="source_a",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="unique conversation text"),
            Message(role="assistant", content="indeed"),
        ],
    )
    conv2 = Conversation(
        id="duplicate_1_first",
        source="source_a",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="duplicate conversation text"),
            Message(role="assistant", content="hello"),
        ],
    )
    conv3 = Conversation(
        id="second_unique",
        source="source_b",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="another unique text"),
            Message(role="assistant", content="hi"),
        ],
    )
    conv4 = Conversation(
        id="duplicate_1_second",
        source="source_b",
        metadata=ConversationMetadata(),
        messages=[
            # Normalized structure is identical to duplicate_1_first
            Message(role="user", content="DUPLICATE CONVERSATION TEXT"),
            Message(role="assistant", content="hello\r\n"),
        ],
    )

    input_list = [conv1, conv2, conv3, conv4]

    # Act
    deduplicated, manifest, report_json, report_md = ConversationDeduplicator.deduplicate(
        input_list, dataset_version="2.1.0"
    )

    # Assert
    # 1. Output count and members
    assert len(deduplicated) == 3
    assert deduplicated[0].id == "first_unique"
    assert deduplicated[1].id == "duplicate_1_first"  # Keeps first occurrence
    assert deduplicated[2].id == "second_unique"  # Order preserved

    # 2. Manifest check
    assert manifest["dataset_version"] == "2.1.0"
    assert manifest["total_conversations_before"] == 4
    assert manifest["total_conversations_after"] == 3
    assert manifest["duplicates_removed"] == 1
    assert manifest["duplicate_percentage"] == 25.0
    assert "created_at" in manifest

    # 3. Report JSON check
    assert report_json["total_conversations_before"] == 4
    assert report_json["total_duplicates_found"] == 1
    assert report_json["total_conversations_after"] == 3
    assert len(report_json["removed_duplicates"]) == 1

    dup_item = report_json["removed_duplicates"][0]
    assert dup_item["kept_conversation_id"] == "duplicate_1_first"
    assert dup_item["removed_conversation_id"] == "duplicate_1_second"
    assert dup_item["kept_source"] == "source_a"
    assert dup_item["removed_source"] == "source_b"
    assert "sha256_hash" in dup_item

    # 4. Report MD check
    assert "**Duplicates removed**: 1" in report_md
    assert "| duplicate_1_first | duplicate_1_second | source_a | source_b |" in report_md


def test_deduplicate_empty_and_no_duplicates() -> None:
    # Arrange
    conv1 = Conversation(
        id="c1",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ],
    )
    conv2 = Conversation(
        id="c2",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="bye"),
            Message(role="assistant", content="goodbye"),
        ],
    )

    # Act
    dedup, manifest, _, md_report = ConversationDeduplicator.deduplicate([conv1, conv2])

    # Assert
    assert len(dedup) == 2
    assert manifest["duplicates_removed"] == 0
    assert manifest["duplicate_percentage"] == 0.0
    assert "No duplicate conversations were found." in md_report

    # Empty list check
    dedup_empty, manifest_empty, _, _ = ConversationDeduplicator.deduplicate([])
    assert len(dedup_empty) == 0
    assert manifest_empty["duplicates_removed"] == 0
    assert manifest_empty["duplicate_percentage"] == 0.0
