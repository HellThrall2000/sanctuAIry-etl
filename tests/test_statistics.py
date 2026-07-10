"""Unit tests for the statistics computation and review exporter."""

from __future__ import annotations

import json
import os
import tempfile

from cbt_companion.analytics.review_exporter import ReviewExporter
from cbt_companion.analytics.statistics_tool import StatisticsTool
from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message
from cbt_companion.models.merge_report import MergeReport


def _create_mock_merge_report() -> MergeReport:
    """Helper to create a populated MergeReport for testing."""
    c1 = Conversation(
        id="c1",
        source="dataset_a",
        metadata=ConversationMetadata(context="joy"),
        messages=[
            Message(role="user", content="Hello, I am happy."),
            Message(role="assistant", content="That is great to hear!"),
        ],
    )
    c2 = Conversation(
        id="c2",
        source="dataset_a",
        metadata=ConversationMetadata(context="sadness"),
        messages=[
            Message(role="user", content="I feel terrible today."),
            Message(role="assistant", content="I am sorry. What is wrong?"),
            Message(role="user", content="Everything."),
            Message(role="assistant", content="I am here for you."),
        ],
    )
    c3 = Conversation(
        id="c3",
        source="dataset_b",
        metadata=ConversationMetadata(context="joy"),
        messages=[
            Message(role="user", content="Yay!"),
            Message(role="assistant", content="Wow!"),
        ],
    )
    return MergeReport(
        merged_conversations=[c1, c2, c3],
        total_conversations=3,
        contribution_counts={"dataset_a": 2, "dataset_b": 1},
        dataset_order=["dataset_a", "dataset_b"],
    )


def test_compute_statistics() -> None:
    # Arrange
    merge_report = _create_mock_merge_report()

    # Act
    stats = StatisticsTool.compute_statistics(merge_report)

    # Assert
    assert stats.total_conversations == 3
    assert stats.conversations_per_dataset == {"dataset_a": 2, "dataset_b": 1}
    assert stats.percentage_contribution_per_dataset == {"dataset_a": 66.67, "dataset_b": 33.33}

    # Turns: c1=2, c2=4, c3=2. Sorted turns: 2, 2, 4.
    assert stats.average_turns == 2.67
    assert stats.median_turns == 2.0
    assert stats.minimum_turns == 2
    assert stats.maximum_turns == 4

    # Message lengths:
    # User messages:
    # - "Hello, I am happy." (19 chars)
    # - "I feel terrible today." (22 chars)
    # - "Everything." (11 chars)
    # - "Yay!" (4 chars)
    # Average User = (18 + 22 + 11 + 4) / 4 = 13.75
    # Assistant messages:
    # - "That is great to hear!" (22 chars)
    # - "I am sorry. What is wrong?" (26 chars)
    # - "I am here for you." (18 chars)
    # - "Wow!" (4 chars)
    # Average Assistant = (22 + 26 + 18 + 4) / 4 = 17.5
    # Max length = 26
    assert stats.average_user_message_length == 13.75
    assert stats.average_assistant_message_length == 17.5
    assert stats.maximum_message_length == 26

    # Context distribution (joy=2, sadness=1)
    assert stats.context_distribution == {"joy": 2, "sadness": 1}


def test_generate_sample_determinism() -> None:
    # Arrange
    merge_report = _create_mock_merge_report()

    # Act & Assert: Deterministic seed checks
    sample_1a = StatisticsTool.generate_sample(merge_report, sample_size=2, seed=42)
    sample_1b = StatisticsTool.generate_sample(merge_report, sample_size=2, seed=42)
    assert [c.id for c in sample_1a] == [c.id for c in sample_1b]

    # Different seeds produce potentially different results (if sample size < total)
    sample_2 = StatisticsTool.generate_sample(merge_report, sample_size=2, seed=100)
    # Check that sample sizes are correct
    assert len(sample_1a) == 2
    assert len(sample_2) == 2

    # Check order preservation of indices
    ids_1a = [c.id for c in sample_1a]
    # Check that order matches the original merge list order (e.g. c1 then c2, or c1 then c3)
    original_ids = [c.id for c in merge_report.merged_conversations]
    assert all(
        original_ids.index(ids_1a[i]) < original_ids.index(ids_1a[i + 1])
        for i in range(len(ids_1a) - 1)
    )

    # Requesting a sample size larger than total returns all conversations in original order
    large_sample = StatisticsTool.generate_sample(merge_report, sample_size=10, seed=42)
    assert len(large_sample) == 3
    assert [c.id for c in large_sample] == ["c1", "c2", "c3"]


def test_exporter_file_writing() -> None:
    # Arrange
    merge_report = _create_mock_merge_report()
    stats = StatisticsTool.compute_statistics(merge_report)
    sample = StatisticsTool.generate_sample(merge_report, sample_size=2, seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_json_path = os.path.join(tmpdir, "reports", "statistics.json")
        stats_md_path = os.path.join(tmpdir, "reports", "statistics.md")
        sample_json_path = os.path.join(tmpdir, "review", "review_sample.json")
        sample_md_path = os.path.join(tmpdir, "review", "review_sample.md")

        # Act
        ReviewExporter.export_statistics(stats, stats_json_path, stats_md_path)
        ReviewExporter.export_review_sample(sample, sample_json_path, sample_md_path)

        # Assert JSON outputs exist and are parseable
        assert os.path.exists(stats_json_path)
        with open(stats_json_path, encoding="utf-8") as f:
            stats_data = json.load(f)
            assert stats_data["total_conversations"] == 3
            assert stats_data["conversations_per_dataset"]["dataset_a"] == 2

        assert os.path.exists(sample_json_path)
        with open(sample_json_path, encoding="utf-8") as f:
            sample_data = json.load(f)
            assert len(sample_data) == 2
            assert sample_data[0]["id"] in ("c1", "c2", "c3")

        # Assert Markdown outputs exist and are non-empty
        assert os.path.exists(stats_md_path)
        with open(stats_md_path, encoding="utf-8") as f:
            stats_md = f.read()
            assert "# Dataset Merger Statistics Report" in stats_md
            assert "dataset_a" in stats_md
            assert "dataset_b" in stats_md

        assert os.path.exists(sample_md_path)
        with open(sample_md_path, encoding="utf-8") as f:
            sample_md = f.read()
            assert "# Dataset Review Sample" in sample_md
            assert "Dialogue Turns" in sample_md
