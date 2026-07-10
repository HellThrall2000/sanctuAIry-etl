"""Filter report model representing results of the filtering stage."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cbt_companion.models.conversation import Conversation


class FilterReport(BaseModel):
    """Structured report returned by a dataset filter.

    Attributes:
        num_kept: Total number of conversations kept.
        num_removed: Total number of conversations removed.
        removed_ids: List of conversation IDs that were removed.
        filtered_conversations: List of valid conversations in original order.
    """

    num_kept: int
    num_removed: int
    removed_ids: list[str] = Field(default_factory=list)
    filtered_conversations: list[Conversation] = Field(default_factory=list)
