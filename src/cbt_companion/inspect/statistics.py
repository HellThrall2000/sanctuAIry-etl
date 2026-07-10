"""Statistical analysis and data quality checks for dataset inspection.

This module provides the computational layer for the inspection framework.
It detects conversation structures heuristically and computes quality metrics
without assuming any fixed schema.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, Sequence, Value

logger = logging.getLogger(__name__)

# ── Thresholds ───────────────────────────────────────────────

LONG_MESSAGE_THRESHOLD = 5_000   # characters
SHORT_MESSAGE_THRESHOLD = 10     # characters


# ── Data Models ──────────────────────────────────────────────

@dataclass
class ColumnInfo:
    """Schema descriptor for a single column."""

    name: str
    dtype: str
    is_nested: bool
    nested_detail: str | None = None


@dataclass
class SchemaInfo:
    """Schema descriptor for one dataset split."""

    columns: list[ColumnInfo]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class ConversationStats:
    """Statistics about detected conversation structure."""

    num_conversations: int
    avg_length: float
    median_length: float
    min_length: int
    max_length: int
    grouping_column: str | None
    grouping_method: str  # "id_grouped" | "sequence_column" | "row_level"


@dataclass
class MessageStats:
    """Character-length statistics across text columns."""

    columns_analyzed: list[str]
    avg_char_length: float
    max_char_length: int


@dataclass
class QualityReport:
    """Data quality check results for one split."""

    missing_values: dict[str, int] = field(default_factory=dict)
    empty_strings: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_ids: dict[str, int] = field(default_factory=dict)
    null_messages: dict[str, int] = field(default_factory=dict)
    type_anomalies: dict[str, str] = field(default_factory=dict)
    extremely_long_messages: dict[str, int] = field(default_factory=dict)
    extremely_short_messages: dict[str, int] = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────

def _mean(values: list[int | float]) -> float:
    """Compute arithmetic mean without importing stdlib statistics (name clash)."""
    return sum(values) / len(values) if values else 0.0


def _median(values: list[int | float]) -> float:
    """Compute median without importing stdlib statistics (name clash)."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])


def feature_type_string(feat: Any) -> str:
    """Convert a ``datasets.Feature`` into a human-readable type string."""
    if isinstance(feat, Value):
        return feat.dtype
    if isinstance(feat, Sequence):
        inner = feature_type_string(feat.feature)
        return f"list[{inner}]"
    if isinstance(feat, dict):
        items = ", ".join(f"{k}: {feature_type_string(v)}" for k, v in feat.items())
        return "{" + items + "}"
    # ClassLabel, Audio, Image, etc.
    return type(feat).__name__


def _is_nested(feat: Any) -> bool:
    return isinstance(feat, (Sequence, dict))


def _find_text_columns(split: Dataset) -> list[str]:
    """Return columns whose feature type is a plain string Value."""
    return [
        name
        for name, feat in split.features.items()
        if isinstance(feat, Value) and feat.dtype == "string"
    ]


_ID_PATTERNS = ("conv_id", "conversation_id", "dialog_id", "dialogue_id")


def _find_grouping_column(split: Dataset) -> str | None:
    """Heuristically find the column used to group rows into conversations."""
    lower_map = {c.lower(): c for c in split.column_names}

    # Exact match on known patterns
    for pattern in _ID_PATTERNS:
        if pattern in lower_map:
            return lower_map[pattern]

    # Substring match
    for pattern in _ID_PATTERNS:
        for low, original in lower_map.items():
            if pattern in low:
                return original

    return None


def _find_sequence_columns(split: Dataset) -> list[str]:
    """Return columns whose feature type is Sequence (lists)."""
    return [
        name
        for name, feat in split.features.items()
        if isinstance(feat, Sequence)
    ]


def _extract_texts_from_value(value: Any) -> list[str]:
    """Recursively extract string fragments from a nested structure."""
    texts: list[str] = []
    if isinstance(value, str):
        texts.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            texts.extend(_extract_texts_from_value(v))
    elif isinstance(value, (list, tuple)):
        for item in value:
            texts.extend(_extract_texts_from_value(item))
    return texts


# ── Public API ───────────────────────────────────────────────

def analyze_schema(split: Dataset) -> SchemaInfo:
    """Extract column names, types, and nesting information."""
    columns: list[ColumnInfo] = []
    for name, feat in split.features.items():
        nested = _is_nested(feat)
        columns.append(ColumnInfo(
            name=name,
            dtype=feature_type_string(feat),
            is_nested=nested,
            nested_detail=feature_type_string(feat) if nested else None,
        ))
    return SchemaInfo(columns=columns)


def compute_conversation_stats(split: Dataset) -> ConversationStats:
    """Detect conversation boundaries and compute turn statistics."""
    # Strategy 1 — ID-based grouping (e.g. EmpatheticDialogues: conv_id)
    grouping_col = _find_grouping_column(split)
    if grouping_col is not None:
        logger.info("Conversation grouping: column '%s'", grouping_col)
        counts = Counter(split[grouping_col])
        lengths = list(counts.values())
        return ConversationStats(
            num_conversations=len(counts),
            avg_length=round(_mean(lengths), 2),
            median_length=_median(lengths),
            min_length=min(lengths),
            max_length=max(lengths),
            grouping_column=grouping_col,
            grouping_method="id_grouped",
        )

    # Strategy 2 — Sequence column (e.g. ESConv: dialog list)
    seq_cols = _find_sequence_columns(split)
    if seq_cols:
        col = seq_cols[0]
        logger.info("Conversation grouping: sequence column '%s'", col)
        values = split[col]
        lengths: list[int] = []
        for v in values:
            if isinstance(v, (list, tuple)):
                lengths.append(len(v))
            elif isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    lengths.append(len(parsed) if isinstance(parsed, list) else 1)
                except (json.JSONDecodeError, TypeError):
                    lengths.append(1)
            else:
                lengths.append(1)

        return ConversationStats(
            num_conversations=len(lengths),
            avg_length=round(_mean(lengths), 2),
            median_length=_median(lengths),
            min_length=min(lengths) if lengths else 0,
            max_length=max(lengths) if lengths else 0,
            grouping_column=col,
            grouping_method="sequence_column",
        )

    # Strategy 3 — Row-level fallback (e.g. Mental Health Counseling: each row is a pair)
    text_cols = _find_text_columns(split)
    turns_per_row = max(len(text_cols), 1)
    logger.info(
        "No grouping column detected — treating each row as a conversation "
        "(%d text column(s) as turns).",
        turns_per_row,
    )
    return ConversationStats(
        num_conversations=split.num_rows,
        avg_length=float(turns_per_row),
        median_length=float(turns_per_row),
        min_length=turns_per_row,
        max_length=turns_per_row,
        grouping_column=None,
        grouping_method="row_level",
    )


def compute_message_stats(split: Dataset) -> MessageStats | None:
    """Compute character-length statistics across all text content."""
    all_lengths: list[int] = []

    # Top-level string columns
    text_cols = _find_text_columns(split)
    for col in text_cols:
        for v in split[col]:
            if isinstance(v, str):
                all_lengths.append(len(v))

    # Nested text inside sequence columns
    seq_cols = _find_sequence_columns(split)
    for col in seq_cols:
        for row_val in split[col]:
            for text in _extract_texts_from_value(row_val):
                all_lengths.append(len(text))

    analyzed = text_cols + seq_cols
    if not all_lengths:
        logger.info("No text content found for message statistics.")
        return None

    return MessageStats(
        columns_analyzed=analyzed,
        avg_char_length=round(_mean(all_lengths), 2),
        max_char_length=max(all_lengths),
    )


def compute_quality_report(split: Dataset) -> QualityReport:
    """Run data quality checks on a single split."""
    logger.info("Running quality checks…")
    report = QualityReport()
    text_cols = set(_find_text_columns(split))

    for col_name in split.column_names:
        values = split[col_name]

        # Missing / null values
        null_count = sum(1 for v in values if v is None)
        if null_count:
            report.missing_values[col_name] = null_count

        if col_name in text_cols:
            # Null text messages
            if null_count:
                report.null_messages[col_name] = null_count

            # Empty strings
            empty = sum(
                1 for v in values
                if v is not None and isinstance(v, str) and v.strip() == ""
            )
            if empty:
                report.empty_strings[col_name] = empty

            # Extreme lengths
            long_count = sum(
                1 for v in values
                if isinstance(v, str) and len(v) > LONG_MESSAGE_THRESHOLD
            )
            if long_count:
                report.extremely_long_messages[col_name] = long_count

            short_count = sum(
                1 for v in values
                if isinstance(v, str) and 0 < len(v.strip()) < SHORT_MESSAGE_THRESHOLD
            )
            if short_count:
                report.extremely_short_messages[col_name] = short_count

    # Duplicate rows (via pandas for efficiency)
    try:
        df = split.to_pandas()
        report.duplicate_rows = int(df.duplicated().sum())
    except Exception:
        logger.warning("Could not compute duplicate rows.", exc_info=True)

    # Duplicate IDs in non-grouping ID columns
    grouping_col = _find_grouping_column(split)
    for col in split.column_names:
        low = col.lower()
        if not (low.endswith("_id") or low == "id"):
            continue
        if col == grouping_col:
            continue  # duplicates are expected in the grouping column
        vals = [v for v in split[col] if v is not None]
        total_unique = len(set(vals))
        if total_unique < len(vals):
            report.duplicate_ids[col] = len(vals) - total_unique

    return report
