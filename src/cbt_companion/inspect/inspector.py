"""Dataset inspector — orchestrates schema, statistics, and quality analysis."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from cbt_companion.inspect.statistics import (
    ConversationStats,
    MessageStats,
    QualityReport,
    SchemaInfo,
    analyze_schema,
    compute_conversation_stats,
    compute_message_stats,
    compute_quality_report,
)
from cbt_companion.models.dataset import DatasetConfig
from datasets import DatasetDict, load_dataset

logger = logging.getLogger(__name__)


# ── Result Models ────────────────────────────────────────────

@dataclass
class SplitInspection:
    """Complete inspection results for one dataset split."""

    name: str
    num_rows: int
    schema: SchemaInfo
    conversation_stats: ConversationStats
    message_stats: MessageStats | None
    quality: QualityReport
    samples: list[dict[str, Any]]


@dataclass
class InspectionResult:
    """Aggregate inspection results across all splits of a dataset."""

    dataset_name: str
    dataset_id: str | None
    splits: dict[str, SplitInspection] = field(default_factory=dict)


# ── Inspector ────────────────────────────────────────────────

class DatasetInspector:
    """Loads a dataset and collects structural, statistical, and quality data.

    The inspector is read-only: it never modifies the underlying data.

    Args:
        config: Registry entry describing the dataset to inspect.
        sample_count: Number of random samples to extract per split.
    """

    def __init__(self, config: DatasetConfig, *, sample_count: int = 5) -> None:
        self._config = config
        self._sample_count = sample_count

    def inspect(self) -> InspectionResult:
        """Run the full inspection pipeline and return an :class:`InspectionResult`."""
        logger.info(
            "Loading dataset '%s' (%s)…",
            self._config.name,
            self._config.dataset_id,
        )
        ds = self._load()

        result = InspectionResult(
            dataset_name=self._config.name,
            dataset_id=self._config.dataset_id,
        )

        for split_name in ds:
            split_data = ds[split_name]
            logger.info(
                "Inspecting split '%s' (%d rows)…",
                split_name,
                split_data.num_rows,
            )
            result.splits[split_name] = SplitInspection(
                name=split_name,
                num_rows=split_data.num_rows,
                schema=analyze_schema(split_data),
                conversation_stats=compute_conversation_stats(split_data),
                message_stats=compute_message_stats(split_data),
                quality=compute_quality_report(split_data),
                samples=self._extract_samples(split_data),
            )

        logger.info("Inspection complete for '%s'.", self._config.name)
        return result

    # ── Private ──────────────────────────────────────────────

    def _load(self) -> DatasetDict:
        """Load the dataset from Hugging Face.

        Attempts Parquet-backed loading first (datasets ≥ 5.x default).
        Falls back to ``trust_remote_code=True`` for datasets that require
        a custom builder script.
        """
        try:
            ds = load_dataset(self._config.dataset_id)
        except (RuntimeError, ValueError):
            # Legacy dataset scripts — retry with trust_remote_code
            logger.debug(
                "Parquet load failed for '%s'; retrying with trust_remote_code.",
                self._config.dataset_id,
            )
            ds = load_dataset(self._config.dataset_id, trust_remote_code=True)

        # load_dataset may return a Dataset (single split) or DatasetDict
        if not isinstance(ds, DatasetDict):
            ds = DatasetDict({"train": ds})
        return ds

    def _extract_samples(self, split_data: Any) -> list[dict[str, Any]]:
        """Randomly select sample records from a split."""
        n = min(self._sample_count, split_data.num_rows)
        if n == 0:
            return []
        indices = sorted(random.sample(range(split_data.num_rows), n))
        return [split_data[i] for i in indices]
