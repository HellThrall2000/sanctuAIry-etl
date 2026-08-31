"""Curation report models describing the training-export stage."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CleaningStats(BaseModel):
    """Counters describing text repairs applied during cleaning.

    Attributes:
        messages_seen: Total number of messages inspected.
        messages_changed: Number of messages whose content was rewritten.
        replacements: Per-rule count of applied substitutions.
    """

    messages_seen: int = 0
    messages_changed: int = 0
    replacements: dict[str, int] = Field(default_factory=dict)


class QualityStats(BaseModel):
    """Counters describing conversations dropped by quality rules.

    Attributes:
        num_seen: Number of conversations inspected.
        num_kept: Number of conversations retained.
        drops_by_reason: Per-rule count of dropped conversations.
    """

    num_seen: int = 0
    num_kept: int = 0
    drops_by_reason: dict[str, int] = Field(default_factory=dict)


class CurationReport(BaseModel):
    """Structured summary of the curation stage.

    Attributes:
        cleaning: Text-repair statistics.
        quality: Quality-filter statistics.
        counts_before: Conversations per source before curation.
        counts_after_quality: Conversations per source after quality filtering.
        counts_after_sampling: Conversations per source after per-source caps.
        num_train: Size of the training split.
        num_validation: Size of the validation split.
        seed: Random seed used for sampling and shuffling.
    """

    cleaning: CleaningStats = Field(default_factory=CleaningStats)
    quality: QualityStats = Field(default_factory=QualityStats)
    counts_before: dict[str, int] = Field(default_factory=dict)
    counts_after_quality: dict[str, int] = Field(default_factory=dict)
    counts_after_sampling: dict[str, int] = Field(default_factory=dict)
    num_train: int = 0
    num_validation: int = 0
    seed: int = 42
