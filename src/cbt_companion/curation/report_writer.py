"""Markdown rendering for the curation report."""

from __future__ import annotations

import json
from pathlib import Path

from cbt_companion.models.curation_report import CurationReport


def _source_table(report: CurationReport) -> list[str]:
    """Build the per-source funnel table rows."""
    sources = sorted(
        set(report.counts_before)
        | set(report.counts_after_quality)
        | set(report.counts_after_sampling)
    )
    lines = [
        "| Source | Deduplicated | After Quality | After Rebalancing | Final Share |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    total_final = sum(report.counts_after_sampling.values()) or 1
    for source in sources:
        before = report.counts_before.get(source, 0)
        after_quality = report.counts_after_quality.get(source, 0)
        after_sampling = report.counts_after_sampling.get(source, 0)
        share = after_sampling / total_final * 100
        lines.append(
            f"| `{source}` | {before} | {after_quality} | {after_sampling} | {share:.2f}% |"
        )
    return lines


class CurationReportWriter:
    """Writes the curation report as JSON and Markdown."""

    @staticmethod
    def export(report: CurationReport, json_path: Path, md_path: Path) -> None:
        """Serialise the report to both output formats.

        Args:
            report: The report to render.
            json_path: Destination for the machine-readable report.
            md_path: Destination for the human-readable report.
        """
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(report.model_dump(), fh, indent=2, ensure_ascii=False)

        lines: list[str] = [
            "# Training Curation Report",
            "",
            "Documents how the deduplicated corpus was cleaned, filtered and",
            "rebalanced into the final supervised fine-tuning splits.",
            "",
            "## Summary",
            "",
            f"- **Training conversations**: {report.num_train}",
            f"- **Validation conversations**: {report.num_validation}",
            f"- **Random seed**: {report.seed}",
            "",
            "## Source Funnel",
            "",
        ]
        lines.extend(_source_table(report))

        lines.extend([
            "",
            "## Text Repairs",
            "",
            f"- **Messages inspected**: {report.cleaning.messages_seen}",
            f"- **Messages rewritten**: {report.cleaning.messages_changed}",
            "",
        ])
        if report.cleaning.replacements:
            lines.extend([
                "| Rule | Replacements |",
                "| :--- | ---: |",
            ])
            for rule, count in sorted(
                report.cleaning.replacements.items(), key=lambda kv: kv[1], reverse=True
            ):
                lines.append(f"| `{rule}` | {count} |")
        else:
            lines.append("_No text repairs were required._")

        lines.extend([
            "",
            "## Quality Filtering",
            "",
            f"- **Conversations inspected**: {report.quality.num_seen}",
            f"- **Conversations retained**: {report.quality.num_kept}",
            "",
        ])
        if report.quality.drops_by_reason:
            lines.extend([
                "| Drop Reason | Conversations |",
                "| :--- | ---: |",
            ])
            for reason, count in sorted(
                report.quality.drops_by_reason.items(), key=lambda kv: kv[1], reverse=True
            ):
                lines.append(f"| `{reason}` | {count} |")
        else:
            lines.append("_No conversations were dropped._")

        lines.append("")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines), encoding="utf-8")
