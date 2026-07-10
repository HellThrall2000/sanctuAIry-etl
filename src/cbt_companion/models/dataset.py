"""Data models for dataset registry entries."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, model_validator


class SourceType(StrEnum):
    """Discriminator for where a dataset is fetched from."""

    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class DatasetConfig(BaseModel, frozen=True):
    """Strongly-typed, immutable descriptor for a single dataset source.

    Attributes:
        key: Unique identifier used as the lookup key in the registry.
        name: Human-readable display name for logs and reports.
        source: Whether the dataset comes from Hugging Face or a local file.
        dataset_id: Hugging Face dataset identifier
            (e.g. ``"facebook/empathetic_dialogues"``).
            Required when *source* is ``huggingface``, otherwise ``None``.
        local_path: Path where the dataset is stored locally (relative to project root).
        description: Brief summary of the dataset's content and purpose.
    """

    key: str
    name: str
    source: SourceType
    dataset_id: str | None = None
    local_path: Path | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check_source_fields(self) -> DatasetConfig:
        """Ensure source-specific fields are present."""
        if self.source is SourceType.HUGGINGFACE and not self.dataset_id:
            raise ValueError(
                f"Dataset '{self.key}': source is 'huggingface' but "
                f"'dataset_id' is missing."
            )
        if self.source is SourceType.LOCAL and not self.local_path:
            raise ValueError(
                f"Dataset '{self.key}': source is 'local' but "
                f"'local_path' is missing."
            )
        return self
