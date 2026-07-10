"""Exporter for writing statistics and review samples to JSON and Markdown."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cbt_companion.models.conversation import Conversation
    from cbt_companion.models.statistics_report import StatisticsReport


class ReviewExporter:
    """Handles file exports (JSON and Markdown formats) for statistics and review samples."""

    @staticmethod
    def export_statistics(stats: StatisticsReport, json_path: str, md_path: str) -> None:
        """Export StatisticsReport to JSON and formatted Markdown files.

        Args:
            stats: The StatisticsReport to write.
            json_path: Filepath for the JSON output.
            md_path: Filepath for the Markdown output.
        """
        # Create directories if they do not exist
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)

        # 1. Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(stats.model_dump_json(indent=2))

        # 2. Format and write Markdown
        lines = [
            "# Dataset Merger Statistics Report",
            "",
            "## Overall Summary",
            f"- **Total Conversations**: {stats.total_conversations}",
            f"- **Average Turns**: {stats.average_turns:.2f}",
            f"- **Median Turns**: {stats.median_turns:.1f}",
            f"- **Minimum Turns**: {stats.minimum_turns}",
            f"- **Maximum Turns**: {stats.maximum_turns}",
            "",
            "## Message Metrics",
            f"- **Average User Message Length (chars)**: {stats.average_user_message_length:.2f}",
            (
                f"- **Average Assistant Message Length (chars)**: "
                f"{stats.average_assistant_message_length:.2f}"
            ),
            f"- **Maximum Message Length (chars)**: {stats.maximum_message_length}",
            "",
            "## Dataset Contributions",
            "",
            "| Dataset Key | Conversations | Contribution % |",
            "| :--- | :--- | :--- |",
        ]

        for source in sorted(stats.conversations_per_dataset.keys()):
            count = stats.conversations_per_dataset[source]
            pct = stats.percentage_contribution_per_dataset.get(source, 0.0)
            lines.append(f"| {source} | {count} | {pct:.2f}% |")

        lines.extend([
            "",
            "## Context Distribution",
            "",
            "| Context | Conversations |",
            "| :--- | :--- |",
        ])

        for ctx, count in stats.context_distribution.items():
            lines.append(f"| {ctx} | {count} |")

        lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def export_review_sample(sample: list[Conversation], json_path: str, md_path: str) -> None:
        """Export Conversation sample to JSON structure and formatted Markdown files.

        Args:
            sample: List of Conversation objects.
            json_path: Filepath for the JSON output.
            md_path: Filepath for the Markdown output.
        """
        # Create directories if they do not exist
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)

        # 1. Write JSON
        sample_dicts = [c.model_dump() for c in sample]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_dicts, f, indent=2, ensure_ascii=False)

        # 2. Write Markdown
        lines = [
            "# Dataset Review Sample",
            "",
            (
                f"This sample contains {len(sample)} conversations "
                f"selected deterministically for review."
            ),
            "",
        ]

        for idx, conv in enumerate(sample, 1):
            lines.extend([
                f"## Conversation {idx}: {conv.id}",
                f"- **Source**: {conv.source}",
                f"- **Split**: {conv.metadata.split or 'None'}",
                f"- **Context**: {conv.metadata.context or 'None'}",
                f"- **Original ID**: {conv.metadata.original_id or 'None'}",
                "",
                "### Dialogue Turns",
                "",
            ])

            for msg in conv.messages:
                lines.append(f"**[{msg.role}]**: {msg.content}")
                lines.append("")

            lines.append("---")
            lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
