"""Merge report model representing results of the merging stage."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cbt_companion.models.conversation import Conversation


class MergeReport(BaseModel):
    """Structured report returned by a dataset merger.

    Attributes:
        merged_conversations: List of combined Conversation objects.
        total_conversations: Total number of merged conversations.
        contribution_counts: Dictionary mapping each dataset's source to its count.
        dataset_order: List of dataset source keys in the order they were merged.
    """

    merged_conversations: list[Conversation] = Field(default_factory=list)
    total_conversations: int
    contribution_counts: dict[str, int] = Field(default_factory=dict)
    dataset_order: list[str] = Field(default_factory=list)
