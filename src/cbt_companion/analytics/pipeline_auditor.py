"""Auditor for tracking stage-by-stage counts, validation breakdowns, and examples."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cbt_companion.models.conversation import Conversation
    from cbt_companion.models.validation_report import ValidationReport


class PipelineAuditor:
    """Tracks conversation counts at each stage, validation failure groups, and logs examples."""

    STAGES = ["Loaded", "Parsed", "Normalized", "Validated", "Filtered", "Merged"]

    def __init__(self) -> None:
        # Maps stage_name -> {dataset_id/source -> count}
        self.stage_counts: dict[str, dict[str, int]] = {stage: {} for stage in self.STAGES}
        # Maps generalized_error -> list of failing conversation IDs
        self.error_ids: dict[str, list[str]] = {}
        # Maps generalized_error -> list of representative Conversation objects (max 10)
        self.error_examples: dict[str, list[Conversation]] = {}
        # Track total failures count for validation breakdown percentage
        self.total_errors_count = 0

    def record_stage(self, stage: str, counts: dict[str, int]) -> None:
        """Record conversation counts for a stage.

        Args:
            stage: One of the STAGES.
            counts: Dict mapping dataset key/source to count.
        """
        if stage in self.stage_counts:
            # We copy to avoid side-effects
            for k, v in counts.items():
                self.stage_counts[stage][k] = v

    @staticmethod
    def _generalize_error(error_msg: str) -> str:
        """Map granular validator messages to general categories."""
        if "fewer than 2 messages" in error_msg:
            return "Conversation has fewer than 2 messages"
        if "start with 'user' role" in error_msg:
            return "Conversation does not start with user role"
        if "end with 'assistant' role" in error_msg:
            return "Conversation does not end with assistant role"
        if "invalid role:" in error_msg:
            return "Message has invalid role"
        if "empty or null content" in error_msg:
            return "Message has empty/null content"
        if "exceeds maximum limit" in error_msg and (
            "Conversation length" in error_msg or "turns" in error_msg
        ):
            return "Conversation length exceeds maximum limit"
        if "exceeds maximum limit" in error_msg and (
            "Message at index" in error_msg or "length" in error_msg
        ):
            return "Message length exceeds maximum limit"
        if "Conversation ID is null, empty or missing" in error_msg:
            return "Conversation ID is null/missing"
        if "Conversation source is null, empty or missing" in error_msg:
            return "Conversation source is null/missing"
        return error_msg

    def record_validation_failures(
        self,
        conversations: list[Conversation],
        validation_report: ValidationReport,
    ) -> None:
        """Record validation failures, grouping by category and storing representative examples.

        Args:
            conversations: Input canonical conversations before filtering.
            validation_report: ValidationReport with results.
        """
        for conv in conversations:
            result = validation_report.results.get(conv.id)
            if not result or result.is_valid:
                continue

            for err in result.errors:
                cat = self._generalize_error(err)
                self.total_errors_count += 1

                # Record conversation ID
                if cat not in self.error_ids:
                    self.error_ids[cat] = []
                self.error_ids[cat].append(conv.id)

                # Record representative example (limit to 10)
                if cat not in self.error_examples:
                    self.error_examples[cat] = []
                if len(self.error_examples[cat]) < 10:
                    # Check if already added to examples (avoid duplicates for same category)
                    if not any(c.id == conv.id for c in self.error_examples[cat]):
                        self.error_examples[cat].append(conv)

    def export_audit(
        self,
        audit_path: str,
        breakdown_path: str,
        examples_dir: str,
    ) -> None:
        """Export all audit reports to disk.

        Args:
            audit_path: Path to pipeline_audit.md.
            breakdown_path: Path to validation_breakdown.md.
            examples_dir: Path to outputs/review/validation_examples/ folder.
        """
        # Create directories
        os.makedirs(os.path.dirname(os.path.abspath(audit_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(breakdown_path)), exist_ok=True)
        os.makedirs(os.path.abspath(examples_dir), exist_ok=True)

        # 1. Export pipeline_audit.md
        self._export_pipeline_audit(audit_path)

        # 2. Export validation_breakdown.md
        self._export_validation_breakdown(breakdown_path)

        # 3. Export validation examples
        self._export_validation_examples(examples_dir)

    def _export_pipeline_audit(self, path: str) -> None:
        # Determine all datasets ever recorded
        all_datasets = set()
        for counts in self.stage_counts.values():
            all_datasets.update(counts.keys())
        sorted_datasets = sorted(list(all_datasets))

        lines = [
            "# Pipeline Audit Report",
            "",
            (
                "This report documents conversation counts and data loss "
                "percentages at each stage of the pipeline."
            ),
            "",
            "## Stage-by-Stage Summary Table",
            "",
        ]

        # Build table headers
        headers = ["Stage"]
        alignments = [":---"]
        for ds in sorted_datasets:
            # use baseline name for shorter column headers
            short_name = ds.split("/")[-1]
            headers.extend([f"{short_name} Count", "% Loss"])
            alignments.extend([":---:", ":---:"])
        headers.extend(["Total Count", "% Loss"])
        alignments.extend([":---:", ":---:"])

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(alignments) + " |")

        for idx, stage in enumerate(self.STAGES):
            row = [stage]
            total_count = sum(self.stage_counts[stage].get(ds, 0) for ds in sorted_datasets)

            # Compute totals loss %
            if idx == 0:
                total_loss_str = "-"
            else:
                prev_stage = self.STAGES[idx - 1]
                prev_total = sum(self.stage_counts[prev_stage].get(ds, 0) for ds in sorted_datasets)
                if prev_total > 0:
                    loss = ((prev_total - total_count) / prev_total) * 100.0
                    total_loss_str = f"{loss:.2f}%"
                else:
                    total_loss_str = "0.00%"

            # Add dataset counts and loss percentages
            for ds in sorted_datasets:
                count = self.stage_counts[stage].get(ds, 0)
                if idx == 0:
                    loss_str = "-"
                else:
                    prev_stage = self.STAGES[idx - 1]
                    prev_count = self.stage_counts[prev_stage].get(ds, 0)
                    if prev_count > 0:
                        loss = ((prev_count - count) / prev_count) * 100.0
                        loss_str = f"{loss:.2f}%"
                    else:
                        loss_str = "0.00%"
                row.extend([str(count), loss_str])

            row.extend([str(total_count), total_loss_str])
            lines.append("| " + " | ".join(row) + " |")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _export_validation_breakdown(self, path: str) -> None:
        lines = [
            "# Validation Failures Breakdown Report",
            "",
            "This report analyzes the distribution of validation failures across the dataset.",
            "",
            "## Summary of Validation Error Categories",
            "",
            "| Error Category | Failures Count | % of All Failures |",
            "| :--- | :---: | :---: |",
        ]

        total = self.total_errors_count if self.total_errors_count > 0 else 1
        sorted_errors = sorted(self.error_ids.items(), key=lambda x: len(x[1]), reverse=True)

        for cat, ids in sorted_errors:
            count = len(ids)
            pct = (count / total) * 100.0
            lines.append(f"| {cat} | {count} | {pct:.2f}% |")

        lines.extend([
            "",
            "## Detailed Conversations Breakdown per Category",
            "",
        ])

        for cat, ids in sorted_errors:
            lines.extend([
                f"### Category: {cat}",
                f"- **Total Failures**: {len(ids)}",
                "- **Failing Conversation IDs**:",
            ])
            for cid in ids:
                lines.append(f"  - `{cid}`")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _export_validation_examples(self, dir_path: str) -> None:
        for cat, examples in self.error_examples.items():
            # Create a clean file name
            safe_name = cat.lower().replace(" ", "_").replace("/", "_").replace("'", "")
            file_path = os.path.join(dir_path, f"{safe_name}.md")

            lines = [
                f"# Validation Examples: {cat}",
                "",
            (
                f"Below are representative examples of conversations "
                f"that failed the validation check: `{cat}`."
            ),
                "",
            ]

            for idx, conv in enumerate(examples, 1):
                lines.extend([
                    f"## Example {idx}: {conv.id}",
                    f"- **Source**: {conv.source}",
                    f"- **Split**: {conv.metadata.split or 'None'}",
                    f"- **Context**: {conv.metadata.context or 'None'}",
                    "",
                    "### Dialogue Turns",
                    "",
                ])

                for msg in conv.messages:
                    lines.append(f"**[{msg.role}]**: {msg.content}")
                    lines.append("")

                lines.append("---")
                lines.append("")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
