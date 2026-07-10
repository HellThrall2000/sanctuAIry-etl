"""Unit tests for the MentalHealthParser."""

from __future__ import annotations

import pytest

from cbt_companion.models.raw_conversation import ParseReport
from cbt_companion.parsers.mental_health_parser import MentalHealthParser


def test_mental_health_parser_success() -> None:
    # Arrange: Mock a list of dataset rows representing mental health counseling logs
    mock_data = [
        {
            "Context": "I feel anxious about exams.",
            "Response": (
                "It is normal to feel that way. "
                "Try breaking down your study material."
            ),
        },
        {
            "Context": "I am having trouble sleeping.",
            "Response": (
                "Establishing a bedtime routine can help "
                "improve your sleep hygiene."
            ),
        }
    ]

    parser = MentalHealthParser()

    # Act
    report = parser.parse(mock_data, split_name="train")

    # Assert
    assert isinstance(report, ParseReport)
    assert report.parser_info.parser_name == "MentalHealthParser"
    assert report.parser_info.dataset_id == "Amod/mental_health_counseling_conversations"
    assert report.split == "train"
    assert report.num_conversations == 2
    assert report.avg_turns == pytest.approx(2.0)
    assert report.max_turns == 2
    assert report.min_turns == 2
    assert len(report.conversations) == 2

    # Verify first conversation
    c1 = report.conversations[0]
    assert c1.id == "mental_health_train_0"
    assert c1.source == "Amod/mental_health_counseling_conversations"
    assert len(c1.messages) == 2

    assert c1.messages[0].content == "I feel anxious about exams."
    assert c1.messages[0].speaker_id == "Context"
    assert c1.messages[0].attributes == {}

    assert c1.messages[1].content == (
        "It is normal to feel that way. "
        "Try breaking down your study material."
    )
    assert c1.messages[1].speaker_id == "Response"
    assert c1.messages[1].attributes == {}

    assert c1.attributes["Context"] == "I feel anxious about exams."
    assert c1.attributes["Response"] == (
        "It is normal to feel that way. "
        "Try breaking down your study material."
    )
    assert c1.attributes["split"] == "train"

    # Verify second conversation
    c2 = report.conversations[1]
    assert c2.id == "mental_health_train_1"
    assert len(c2.messages) == 2
    assert c2.messages[0].content == "I am having trouble sleeping."
    assert c2.messages[0].speaker_id == "Context"
    assert c2.messages[1].content == (
        "Establishing a bedtime routine can help "
        "improve your sleep hygiene."
    )
    assert c2.messages[1].speaker_id == "Response"


def test_mental_health_parser_empty_context() -> None:
    bad_data = [
        {
            "Context": "",
            "Response": "I am here to help.",
        }
    ]
    parser = MentalHealthParser()

    report = parser.parse(bad_data, "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "empty or missing 'Context'" in report.warnings[0]


def test_mental_health_parser_empty_response() -> None:
    bad_data = [
        {
            "Context": "Hello",
            "Response": "  ",
        }
    ]
    parser = MentalHealthParser()

    report = parser.parse(bad_data, "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 0
    assert len(report.warnings) == 1
    assert "empty or missing 'Response'" in report.warnings[0]


def test_mental_health_parser_missing_fields() -> None:
    bad_data = [
        {
            "Context": "Hello",
            "Response": "Hi",
        },
        {
            "Context": "Hello2",
            # Response is missing completely
        }
    ]
    parser = MentalHealthParser()

    report = parser.parse(bad_data, "train")

    assert report.skipped_conversations == 1
    assert len(report.conversations) == 1
    assert len(report.warnings) == 1
    assert "empty or missing 'Response'" in report.warnings[0]


def test_mental_health_parser_dataset_level_failure() -> None:
    bad_schema = [
        {
            "Context": "Hello",
            # missing Response column completely
        }
    ]
    parser = MentalHealthParser()

    with pytest.raises(ValueError) as excinfo:
        parser.parse(bad_schema, "train")
    assert "Missing required column(s)" in str(excinfo.value)

