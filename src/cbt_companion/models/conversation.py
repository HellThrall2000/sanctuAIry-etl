"""Canonical conversation models for the normalized dataset pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    """A single dialogue turn/message in a normalized conversation.

    Attributes:
        role: Mapped role of the speaker (strictly "user" or "assistant").
        content: Message content text.
    """

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty or whitespace-only.")
        return v


class ConversationMetadata(BaseModel):
    """Canonical metadata for a conversation.

    Attributes:
        split: The dataset split (e.g. "train", "validation").
        source_dataset: The Hugging Face dataset identifier.
        original_id: Original identifier from the source dataset.
        context: Context description or category (e.g. emotion or situation).
    """

    split: str | None = None
    source_dataset: str | None = None
    original_id: str | None = None
    context: str | None = None


class Conversation(BaseModel):
    """Canonical conversation model.

    Attributes:
        id: Canonical conversation ID.
        source: Key or identifier of the source dataset.
        metadata: Normalized conversation metadata.
        messages: List of message turns in chronological order.
    """

    id: str
    source: str
    metadata: ConversationMetadata
    messages: list[Message]

    @field_validator("id", "source")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace-only.")
        return v

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("Conversation must have at least one message.")
        return v
