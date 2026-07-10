"""Normalizer for the ESConv dataset."""

from __future__ import annotations

from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message
from cbt_companion.models.raw_conversation import RawConversation
from cbt_companion.normalizers.base import BaseNormalizer


class ESConvNormalizer(BaseNormalizer):
    """Normalizes ESConv raw conversations."""

    def normalize(self, raw_conv: RawConversation) -> Conversation:
        """Convert ESConv RawConversation to canonical Conversation."""
        messages = []
        for msg in raw_conv.messages:
            role = "user" if msg.speaker_id == "usr" else "assistant"
            messages.append(
                Message(
                    role=role,
                    content=msg.content,
                )
            )

        metadata = ConversationMetadata(
            split=raw_conv.attributes.get("split"),
            source_dataset="thu-coai/esconv",
            original_id=raw_conv.id,
            context=raw_conv.attributes.get("emotion_type"),
        )

        return Conversation(
            id=raw_conv.id,
            source="thu-coai/esconv",
            metadata=metadata,
            messages=messages,
        )
