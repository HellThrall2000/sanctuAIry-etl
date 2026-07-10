"""Markdown report generator for dataset inspection results."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cbt_companion.inspect.inspector import InspectionResult, SplitInspection

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> str:
    """Fallback serializer for non-standard types."""
    return str(obj)


class MarkdownReportGenerator:
    """Renders an :class:`InspectionResult` as a Markdown document.

    Args:
        result: The completed inspection result to render.
    """

    def __init__(self, result: InspectionResult) -> None:
        self._result = result

    # ── Public ───────────────────────────────────────────────

    def generate(self) -> str:
        """Build the full Markdown report string."""
        r = self._result
        lines: list[str] = []

        # Header
        lines.append(f"# Inspection Report: {r.dataset_name}\n")
        lines.append(f"**Hugging Face ID:** `{r.dataset_id}`  ")
        lines.append(
            f"**Inspection Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}  "
        )
        split_names = ", ".join(f"`{s}`" for s in r.splits)
        lines.append(f"**Splits:** {split_names}\n")

        # Per-split sections
        for split_name, split_data in r.splits.items():
            lines.append("---\n")
            lines.append(
                f"## Split: `{split_name}` — {split_data.num_rows:,} rows\n"
            )
            lines.extend(self._render_schema(split_data))
            lines.extend(self._render_conversation_stats(split_data))
            lines.extend(self._render_message_stats(split_data))
            lines.extend(self._render_quality(split_data))
            lines.extend(self._render_samples(split_data))

        return "\n".join(lines) + "\n"

    def save(self, output_dir: Path) -> Path:
        """Write the report to ``output_dir/<dataset_key>.md``.

        Returns:
            The path to the saved report file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Derive filename from dataset_id or name
        dataset_id = self._result.dataset_id or self._result.dataset_name
        filename = dataset_id.replace("/", "_").replace(" ", "_").lower() + ".md"
        filepath = output_dir / filename

        content = self.generate()
        filepath.write_text(content, encoding="utf-8")
        logger.info("Report saved: %s", filepath)
        return filepath

    # ── Private Section Renderers ────────────────────────────

    def _render_schema(self, split: SplitInspection) -> list[str]:
        lines: list[str] = [
            "### Schema\n",
            "| # | Column | Type | Nested |",
            "|---|--------|------|--------|",
        ]
        for idx, col in enumerate(split.schema.columns, start=1):
            nested_str = f"Yes — `{col.nested_detail}`" if col.is_nested else "No"
            lines.append(f"| {idx} | `{col.name}` | `{col.dtype}` | {nested_str} |")
        lines.append("")
        return lines

    def _render_conversation_stats(self, split: SplitInspection) -> list[str]:
        cs = split.conversation_stats
        lines: list[str] = [
            "### Conversation Structure\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Detection method | `{cs.grouping_method}` |",
            f"| Grouping column | `{cs.grouping_column or 'N/A'}` |",
            f"| Conversations | {cs.num_conversations:,} |",
            f"| Avg turns per conversation | {cs.avg_length} |",
            f"| Median turns | {cs.median_length} |",
            f"| Min turns | {cs.min_length} |",
            f"| Max turns | {cs.max_length} |",
            "",
        ]
        return lines

    def _render_message_stats(self, split: SplitInspection) -> list[str]:
        ms = split.message_stats
        if ms is None:
            return ["### Message Statistics\n", "_No text columns detected._\n"]

        cols = ", ".join(f"`{c}`" for c in ms.columns_analyzed)
        lines: list[str] = [
            "### Message Statistics\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Columns analyzed | {cols} |",
            f"| Avg character length | {ms.avg_char_length:,.1f} |",
            f"| Max character length | {ms.max_char_length:,} |",
            "",
        ]
        return lines

    def _render_quality(self, split: SplitInspection) -> list[str]:
        q = split.quality
        lines: list[str] = ["### Data Quality\n"]

        has_issues = any([
            q.missing_values, q.empty_strings, q.duplicate_rows,
            q.duplicate_ids, q.null_messages,
            q.extremely_long_messages, q.extremely_short_messages,
        ])

        if not has_issues:
            lines.append("_No quality issues detected._\n")
            return lines

        lines.extend([
            "| Check | Column | Count |",
            "|-------|--------|-------|",
        ])

        for col, cnt in q.missing_values.items():
            lines.append(f"| Missing values | `{col}` | {cnt:,} |")

        for col, cnt in q.empty_strings.items():
            lines.append(f"| Empty strings | `{col}` | {cnt:,} |")

        if q.duplicate_rows:
            lines.append(f"| Duplicate rows | _all columns_ | {q.duplicate_rows:,} |")

        for col, cnt in q.duplicate_ids.items():
            lines.append(f"| Duplicate IDs | `{col}` | {cnt:,} |")

        for col, cnt in q.null_messages.items():
            lines.append(f"| Null messages | `{col}` | {cnt:,} |")

        for col, cnt in q.extremely_long_messages.items():
            lines.append(f"| Extremely long (>{_LONG_THR}ch) | `{col}` | {cnt:,} |")

        for col, cnt in q.extremely_short_messages.items():
            lines.append(f"| Extremely short (<{_SHORT_THR}ch) | `{col}` | {cnt:,} |")

        lines.append("")
        return lines

    def _render_samples(self, split: SplitInspection) -> list[str]:
        lines: list[str] = [
            f"### Sample Records ({len(split.samples)})\n",
        ]
        if not split.samples:
            lines.append("_No samples available._\n")
            return lines

        for idx, record in enumerate(split.samples, start=1):
            lines.append(f"#### Sample {idx}\n")
            lines.append("```json")
            lines.append(json.dumps(record, indent=2, default=_json_default, ensure_ascii=False))
            lines.append("```\n")
        return lines


# ── Module-level threshold imports for the report template ───
from cbt_companion.inspect.statistics import (  # noqa: E402
    LONG_MESSAGE_THRESHOLD as _LONG_THR,
)
from cbt_companion.inspect.statistics import (
    SHORT_MESSAGE_THRESHOLD as _SHORT_THR,
)
