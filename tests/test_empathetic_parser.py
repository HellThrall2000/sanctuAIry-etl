"""Unit tests for the EmpatheticParser."""

from __future__ import annotations

import pytest

from cbt_companion.models.raw_conversation import ParseReport
from cbt_companion.parsers.empathetic_parser import EmpatheticParser


def test_empathetic_parser_success() -> None:
    # Arrange: Mock a list of dataset rows representing two conversations
    mock_data = [
        # Conversation 1
        {
            "conv_id": "conv_1",
            "utterance_idx": 2,
            "speaker_idx": 101,
            "utterance": "Hello, how are you?",
            "context": "nostalgic",
            "prompt": "I was thinking about childhood.",
            "selfeval": "5|5",
            "tags": "tag_a",
        },
        {
            "conv_id": "conv_1",
            "utterance_idx": 1,
            "speaker_idx": 100,
            "utterance": "Hi!",
            "context": "nostalgic",
            "prompt": "I was thinking about childhood.",
            "selfeval": "5|5",
            "tags": "tag_b",
        },
        # Conversation 2
        {
            "conv_id": "conv_2",
            "utterance_idx": 1,
            "speaker_idx": "spk_abc",
            "utterance": "I feel lonely.",
            "context": "lonely",
            "prompt": "Nobody called me today.",
            "selfeval": "4|4",
            "tags": "",
        },
        {
            "conv_id": "conv_2",
            "utterance_idx": 2,
            "speaker_idx": "spk_xyz",
            "utterance": "I'm here for you.",
            "context": "lonely",
            "prompt": "Nobody called me today.",
            "selfeval": "5|5",
            "tags": "empathetic",
        },
        {
            "conv_id": "conv_2",
            "utterance_idx": 3,
            "speaker_idx": "spk_abc",
            "utterance": "Thank you.",
            "context": "lonely",
            "prompt": "Nobody called me today.",
            "selfeval": "5|5",
            "tags": "",
        },
    ]

    parser = EmpatheticParser()

    # Act: Parse the mock dataset split
    report = parser.parse(mock_data, split_name="validation")

    # Assert: ParseReport attributes
    assert isinstance(report, ParseReport)
    assert report.parser_info.parser_name == "EmpatheticParser"
    assert report.parser_info.dataset_id == "facebook/empathetic_dialogues"
    assert report.split == "validation"
    assert report.num_conversations == 2
    assert report.avg_turns == pytest.approx(2.5)  # conv_1 has 2, conv_2 has 3
    assert report.max_turns == 3
    assert report.min_turns == 2
    assert len(report.conversations) == 2

    # Verify conversation order and grouping
    conv_map = {c.id: c for c in report.conversations}
    assert "conv_1" in conv_map
    assert "conv_2" in conv_map

    # Assert: conv_1 details (should be sorted by utterance_idx, meaning "Hi!" comes first)
    c1 = conv_map["conv_1"]
    assert c1.id == "conv_1"
    assert c1.source == "facebook/empathetic_dialogues"
    assert len(c1.messages) == 2

    assert c1.messages[0].content == "Hi!"
    assert c1.messages[0].speaker_id == 100  # Native integer type preserved
    assert c1.messages[0].attributes["selfeval"] == "5|5"
    assert c1.messages[0].attributes["tags"] == "tag_b"
    assert c1.messages[0].attributes["utterance_idx"] == 1

    assert c1.messages[1].content == "Hello, how are you?"
    assert c1.messages[1].speaker_id == 101  # Native integer type preserved
    assert c1.messages[1].attributes["utterance_idx"] == 2

    assert c1.attributes["context"] == "nostalgic"
    assert c1.attributes["prompt"] == "I was thinking about childhood."
    assert c1.attributes["split"] == "validation"

    # Assert: conv_2 details
    c2 = conv_map["conv_2"]
    assert len(c2.messages) == 3
    assert c2.messages[0].speaker_id == "spk_abc"  # Native string type preserved
    assert c2.messages[1].speaker_id == "spk_xyz"
    assert c2.messages[2].speaker_id == "spk_abc"
    assert c2.messages[0].content == "I feel lonely."
    assert c2.messages[1].content == "I'm here for you."
    assert c2.messages[2].content == "Thank you."


def test_empathetic_parser_duplicate_or_out_of_order_utterance_idx() -> None:
    # Arrange: duplicate utterance_idx within same conv_id
    bad_data = [
        {
            "conv_id": "conv_bad",
            "utterance_idx": 1,
            "speaker_idx": 1,
            "utterance": "First",
        },
        {
            "conv_id": "conv_bad",
            "utterance_idx": 1,  # Duplicate index
            "speaker_idx": 2,
            "utterance": "Duplicate",
        },
    ]

    parser = EmpatheticParser()

    # Act
    report = parser.parse(bad_data, "train")

    # Assert
    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "out-of-order indices" in report.warnings[0]


def test_empathetic_parser_empty_utterance() -> None:
    bad_data = [
        {
            "conv_id": "conv_bad",
            "utterance_idx": 1,
            "speaker_idx": 1,
            "utterance": "",  # Empty content
        }
    ]

    parser = EmpatheticParser()

    report = parser.parse(bad_data, "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "empty 'utterance'" in report.warnings[0]


def test_empathetic_parser_missing_speaker() -> None:
    bad_data = [
        {
            "conv_id": "conv_bad",
            "utterance_idx": 1,
            "speaker_idx": None,  # Missing speaker
            "utterance": "Hello",
        }
    ]

    parser = EmpatheticParser()

    report = parser.parse(bad_data, "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "missing/empty speaker_idx" in report.warnings[0]


def test_empathetic_parser_missing_conv_id() -> None:
    bad_data = [
        {
            "conv_id": "",  # Empty conv_id
            "utterance_idx": 1,
            "speaker_idx": 1,
            "utterance": "Hello",
        }
    ]

    parser = EmpatheticParser()

    report = parser.parse(bad_data, "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "missing a valid 'conv_id'" in report.warnings[0]


def test_empathetic_parser_dataset_level_failure() -> None:
    # missing required column: speaker_idx
    bad_schema_data = [
        {
            "conv_id": "conv_1",
            "utterance_idx": 1,
            "utterance": "Hello",
        }
    ]

    parser = EmpatheticParser()

    with pytest.raises(ValueError) as excinfo:
        parser.parse(bad_schema_data, "train")
    assert "Missing required column(s)" in str(excinfo.value)

