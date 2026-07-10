"""Unit tests for the pipeline auditing framework."""

from __future__ import annotations

import os
import tempfile

from cbt_companion.analytics.pipeline_auditor import PipelineAuditor
from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message
from cbt_companion.models.validation_report import ConversationValidationResult, ValidationReport


def test_pipeline_auditor_stage_recording() -> None:
    # Arrange
    auditor = PipelineAuditor()

    # Act
    auditor.record_stage("Loaded", {"dataset_a": 100, "dataset_b": 50})
    auditor.record_stage("Parsed", {"dataset_a": 100, "dataset_b": 40})

    # Assert
    assert auditor.stage_counts["Loaded"] == {"dataset_a": 100, "dataset_b": 50}
    assert auditor.stage_counts["Parsed"] == {"dataset_a": 100, "dataset_b": 40}


def test_pipeline_auditor_validation_failures() -> None:
    # Arrange: Setup mock conversations and validation report with failures
    c1 = Conversation(
        id="c1",
        source="src",
        metadata=ConversationMetadata(),
        messages=[Message(role="user", content="Hello")],
    )
    c2 = Conversation(
        id="c2",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="assistant", content="Hello"),
            Message(role="user", content="Hi"),
        ],
    )

    val_report = ValidationReport(
        num_checked=2,
        num_valid=0,
        num_invalid=2,
        results={
            "c1": ConversationValidationResult(
                conversation_id="c1",
                is_valid=False,
                errors=["Conversation has fewer than 2 messages (got 1)."],
            ),
            "c2": ConversationValidationResult(
                conversation_id="c2",
                is_valid=False,
                errors=[
                    "Message at index 0 has invalid role: 'invalid_role'.",
                    "Message at index 1 has empty or null content.",
                ],
            ),
        },
    )

    auditor = PipelineAuditor()

    # Act
    auditor.record_validation_failures([c1, c2], val_report)

    # Assert generalized keys
    assert "Conversation has fewer than 2 messages" in auditor.error_ids
    assert "Message has invalid role" in auditor.error_ids
    assert "Message has empty/null content" in auditor.error_ids

    # Assert failing conversation IDs are recorded
    assert auditor.error_ids["Conversation has fewer than 2 messages"] == ["c1"]
    assert auditor.error_ids["Message has invalid role"] == ["c2"]
    assert auditor.error_ids["Message has empty/null content"] == ["c2"]

    # Assert representative examples are captured
    assert len(auditor.error_examples["Conversation has fewer than 2 messages"]) == 1
    assert auditor.error_examples["Conversation has fewer than 2 messages"][0].id == "c1"


def test_auditor_reports_export() -> None:
    # Arrange
    auditor = PipelineAuditor()
    auditor.record_stage("Loaded", {"dataset_a": 100, "dataset_b": 50})
    auditor.record_stage("Parsed", {"dataset_a": 100, "dataset_b": 50})
    auditor.record_stage("Normalized", {"dataset_a": 100, "dataset_b": 50})
    auditor.record_stage("Validated", {"dataset_a": 100, "dataset_b": 50})
    auditor.record_stage("Filtered", {"dataset_a": 90, "dataset_b": 40})
    auditor.record_stage("Merged", {"dataset_a": 90, "dataset_b": 40})

    c1 = Conversation(
        id="c1",
        source="dataset_a",
        metadata=ConversationMetadata(),
        messages=[Message(role="user", content="Hello")],
    )
    val_report = ValidationReport(
        num_checked=1,
        num_valid=0,
        num_invalid=1,
        results={
            "c1": ConversationValidationResult(
                conversation_id="c1",
                is_valid=False,
                errors=["Conversation has fewer than 2 messages (got 1)."],
            )
        },
    )
    auditor.record_validation_failures([c1], val_report)

    with tempfile.TemporaryDirectory() as tmpdir:
        audit_path = os.path.join(tmpdir, "pipeline_audit.md")
        breakdown_path = os.path.join(tmpdir, "validation_breakdown.md")
        examples_dir = os.path.join(tmpdir, "validation_examples")

        # Act
        auditor.export_audit(audit_path, breakdown_path, examples_dir)

        # Assert files are written
        assert os.path.exists(audit_path)
        assert os.path.exists(breakdown_path)
        assert os.path.exists(examples_dir)

        # Assert contents of pipeline_audit.md
        with open(audit_path, encoding="utf-8") as f:
            content = f.read()
            assert "# Pipeline Audit Report" in content
            assert "| Stage |" in content
            assert "dataset_a" in content
            assert "dataset_b" in content
            # check data loss percentage exists
            assert "10.00%" in content  # dataset_a loss: (100 - 90)/100 = 10%
            assert "20.00%" in content  # dataset_b loss: (50 - 40)/50 = 20%

        # Assert contents of validation_breakdown.md
        with open(breakdown_path, encoding="utf-8") as f:
            breakdown = f.read()
            assert "# Validation Failures Breakdown Report" in breakdown
            assert "Conversation has fewer than 2 messages" in breakdown
            # check IDs list
            assert "Failing Conversation IDs" in breakdown
            assert "- `c1`" in breakdown

        # Assert contents of validation_examples files
        example_file = os.path.join(examples_dir, "conversation_has_fewer_than_2_messages.md")
        assert os.path.exists(example_file)
        with open(example_file, encoding="utf-8") as f:
            examples_content = f.read()
            expected_title = "# Validation Examples: Conversation has fewer than 2 messages"
            assert expected_title in examples_content
            assert "Example 1: c1" in examples_content
