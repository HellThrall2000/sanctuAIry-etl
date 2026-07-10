"""Raw conversation models and parse reporting schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawMessage(BaseModel):
    """A single turn/message in a conversation preserving raw dataset values.

    Attributes:
        speaker_id: Native speaker identifier (e.g. integer or string).
        content: The raw text content of the message.
        attributes: Raw, source-specific message attributes (e.g. selfeval, tags).
    """

    speaker_id: int | str
    content: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class RawConversation(BaseModel):
    """Canonical raw conversation representation.

    Attributes:
        id: Original conversation identifier from the source dataset.
        source: Key or identifier of the source dataset.
        messages: Chronologically ordered list of raw message turns.
        attributes: Raw conversation-level attributes (e.g. split, context, emotion).
    """

    id: str
    source: str
    messages: list[RawMessage]
    attributes: dict[str, Any] = Field(default_factory=dict)


class ParserInfo(BaseModel):
    """Metadata describing the parser that generated the report."""

    parser_name: str
    dataset_id: str


class ParseReport(BaseModel):
    """Structured report returned by a parser containing statistics and results.

    Attributes:
        parser_info: Information about the parser used.
        split: The dataset split parsed.
        num_conversations: Number of conversations parsed.
        avg_turns: Average number of turns per conversation.
        max_turns: Maximum number of turns in a single conversation.
        min_turns: Minimum number of turns in a single conversation.
        warnings: List of non-fatal warnings encountered during parsing.
        errors: List of fatal or non-fatal errors encountered during parsing.
        conversations: List of parsed RawConversation objects.
    """

    parser_info: ParserInfo
    split: str
    num_conversations: int
    avg_turns: float
    max_turns: int
    min_turns: int
    skipped_conversations: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    conversations: list[RawConversation] = Field(default_factory=list)

    @classmethod
    def from_conversations(
        cls,
        conversations: list[RawConversation],
        parser_info: ParserInfo,
        split: str,
        skipped_conversations: int = 0,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> ParseReport:
        """Helper to construct a ParseReport and compute statistics from conversations."""
        warnings = warnings or []
        errors = errors or []
        num_conversations = len(conversations)

        if num_conversations == 0:
            return cls(
                parser_info=parser_info,
                split=split,
                num_conversations=0,
                avg_turns=0.0,
                max_turns=0,
                min_turns=0,
                skipped_conversations=skipped_conversations,
                warnings=warnings,
                errors=errors,
                conversations=conversations,
            )

        turns = [len(c.messages) for c in conversations]
        avg_turns = sum(turns) / num_conversations
        max_turns = max(turns)
        min_turns = min(turns)

        return cls(
            parser_info=parser_info,
            split=split,
            num_conversations=num_conversations,
            avg_turns=avg_turns,
            max_turns=max_turns,
            min_turns=min_turns,
            skipped_conversations=skipped_conversations,
            warnings=warnings,
            errors=errors,
            conversations=conversations,
        )
