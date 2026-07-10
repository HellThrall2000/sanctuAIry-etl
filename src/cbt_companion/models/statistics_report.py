"""Statistics report model representing dataset analytics."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StatisticsReport(BaseModel):
    """Structured report returned by the StatisticsTool.

    Attributes:
        total_conversations: Total number of conversations.
        conversations_per_dataset: Count of conversations per dataset key.
        percentage_contribution_per_dataset: Percentage contribution of each dataset.
        average_turns: Average turn count across conversations.
        median_turns: Median turn count across conversations.
        minimum_turns: Minimum turn count in any single conversation.
        maximum_turns: Maximum turn count in any single conversation.
        average_user_message_length: Average character length of user messages.
        average_assistant_message_length: Average character length of assistant messages.
        maximum_message_length: Maximum character length of any message.
        context_distribution: Count of each context category.
    """

    total_conversations: int
    conversations_per_dataset: dict[str, int] = Field(default_factory=dict)
    percentage_contribution_per_dataset: dict[str, float] = Field(default_factory=dict)
    average_turns: float
    median_turns: float
    minimum_turns: int
    maximum_turns: int
    average_user_message_length: float
    average_assistant_message_length: float
    maximum_message_length: int
    context_distribution: dict[str, int] = Field(default_factory=dict)
