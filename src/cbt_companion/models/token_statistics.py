"""Data model representing token statistics for fine-tuning datasets."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenStatisticsReport(BaseModel):
    """Pydantic model representing token analysis statistics using a model tokenizer.

    Attributes:
        total_conversations: Total number of conversations analyzed.
        total_tokens: Total tokens across all conversations.
        average_tokens: Average tokens per conversation.
        median_tokens: Median tokens per conversation.
        minimum_tokens: Minimum tokens in a single conversation.
        maximum_tokens: Maximum tokens in a single conversation.
        percentile_95_tokens: 95th percentile of token counts.
        percentile_99_tokens: 99th percentile of token counts.
        average_turns_per_conversation: Average number of message turns per conversation.
        recommended_context_length: The recommended model context size.
        conversations_over_1024: Conversations with token counts > 1024.
        conversations_over_2048: Conversations with token counts > 2048.
        conversations_over_4096: Conversations with token counts > 4096.
        conversations_over_8192: Conversations with token counts > 8192.
        histogram: A mapping from length range bins to conversation count.
    """

    total_conversations: int
    total_tokens: int
    average_tokens: float
    median_tokens: float
    minimum_tokens: int
    maximum_tokens: int
    percentile_95_tokens: int
    percentile_99_tokens: int
    average_turns_per_conversation: float
    recommended_context_length: int
    conversations_over_1024: int
    conversations_over_2048: int
    conversations_over_4096: int
    conversations_over_8192: int
    histogram: dict[str, int] = Field(default_factory=dict)
