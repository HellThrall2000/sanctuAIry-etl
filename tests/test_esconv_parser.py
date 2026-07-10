"""Unit tests for the ESConvParser."""

from __future__ import annotations

import json

import pytest

from cbt_companion.models.raw_conversation import ParseReport
from cbt_companion.parsers.esconv_parser import ESConvParser


def test_esconv_parser_success() -> None:
    # Arrange: Mock text fields containing JSON strings representing conversations
    conv_1_data = {
        "experience_type": "Current Experience",
        "emotion_type": "sadness",
        "problem_type": "breakup",
        "situation": "I broke up with my partner.",
        "survey_score": {
            "seeker": {"initial": "5", "final": "3"}
        },
        "dialog": [
            {"text": "Hi", "speaker": "usr"},
            {"text": "Hello, how can I help?", "speaker": "sys", "strategy": "Question"},
            {"text": "I am sad.", "speaker": "usr"}
        ],
        "seeker_question1": "Yes",
    }

    conv_2_data = {
        "experience_type": "Previous Experience",
        "emotion_type": "anxiety",
        "problem_type": "job",
        "situation": "Stressed about interview.",
        "survey_score": {
            "seeker": {"initial": "4", "final": "2"}
        },
        "dialog": [
            {"text": "Hello", "speaker": "usr"},
            {"text": "Hey!", "speaker": "sys", "strategy": "Others"}
        ],
        "seeker_question1": "No",
    }

    mock_dataset = [
        {"text": json.dumps(conv_1_data)},
        {"text": json.dumps(conv_2_data)}
    ]

    parser = ESConvParser()

    # Act
    report = parser.parse(mock_dataset, split_name="train")

    # Assert
    assert isinstance(report, ParseReport)
    assert report.parser_info.parser_name == "ESConvParser"
    assert report.parser_info.dataset_id == "thu-coai/esconv"
    assert report.split == "train"
    assert report.num_conversations == 2
    assert report.avg_turns == pytest.approx(2.5)  # 3 turns in conv_1, 2 turns in conv_2
    assert report.max_turns == 3
    assert report.min_turns == 2
    assert len(report.conversations) == 2

    # Verify first conversation
    c1 = report.conversations[0]
    assert c1.id == "esconv_train_0"
    assert c1.source == "thu-coai/esconv"
    assert len(c1.messages) == 3

    assert c1.messages[0].content == "Hi"
    assert c1.messages[0].speaker_id == "usr"
    assert c1.messages[0].attributes == {}

    assert c1.messages[1].content == "Hello, how can I help?"
    assert c1.messages[1].speaker_id == "sys"
    assert c1.messages[1].attributes == {"strategy": "Question"}

    assert c1.attributes["experience_type"] == "Current Experience"
    assert c1.attributes["emotion_type"] == "sadness"
    assert c1.attributes["problem_type"] == "breakup"
    assert c1.attributes["situation"] == "I broke up with my partner."
    assert c1.attributes["survey_score"] == {"seeker": {"initial": "5", "final": "3"}}
    assert c1.attributes["seeker_question1"] == "Yes"
    assert c1.attributes["split"] == "train"

    # Verify second conversation
    c2 = report.conversations[1]
    assert c2.id == "esconv_train_1"
    assert len(c2.messages) == 2
    assert c2.messages[0].content == "Hello"
    assert c2.messages[0].speaker_id == "usr"
    assert c2.messages[1].content == "Hey!"
    assert c2.messages[1].speaker_id == "sys"
    assert c2.messages[1].attributes == {"strategy": "Others"}
    assert c2.attributes["emotion_type"] == "anxiety"


def test_esconv_parser_invalid_json() -> None:
    bad_data = [
        {"text": "{ invalid json }"}
    ]
    parser = ESConvParser()

    report = parser.parse(bad_data, "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "invalid JSON" in report.warnings[0]


def test_esconv_parser_missing_dialog() -> None:
    bad_data = [
        {"text": json.dumps({"experience_type": "test"})}  # missing dialog list
    ]
    parser = ESConvParser()

    report = parser.parse(bad_data, "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "missing a valid 'dialog' list" in report.warnings[0]


def test_esconv_parser_empty_text_turn() -> None:
    bad_data = {
        "dialog": [
            {"text": "", "speaker": "usr"}
        ]
    }
    parser = ESConvParser()

    report = parser.parse([{"text": json.dumps(bad_data)}], "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "empty/missing text" in report.warnings[0]


def test_esconv_parser_missing_speaker() -> None:
    bad_data = {
        "dialog": [
            {"text": "Hello", "speaker": None}
        ]
    }
    parser = ESConvParser()

    report = parser.parse([{"text": json.dumps(bad_data)}], "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "missing/empty speaker" in report.warnings[0]


def test_esconv_parser_missing_text_column() -> None:
    bad_data = [
        {"not_text": "data"}
    ]
    parser = ESConvParser()

    with pytest.raises(ValueError) as excinfo:
        parser.parse(bad_data, "train")
    assert "Missing required column: 'text'" in str(excinfo.value)
