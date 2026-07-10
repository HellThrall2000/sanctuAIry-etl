"""Normalizer for the Mental Health Counseling Conversations dataset."""

from __future__ import annotations

from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message
from cbt_companion.models.raw_conversation import RawConversation
from cbt_companion.normalizers.base import BaseNormalizer


class MentalHealthNormalizer(BaseNormalizer):
    """Normalizes Mental Health Counseling Conversations raw conversations."""

    def normalize(self, raw_conv: RawConversation) -> Conversation:
        """Convert Mental Health RawConversation to canonical Conversation."""
        messages = []
        for msg in raw_conv.messages:
            role = "user" if msg.speaker_id == "Context" else "assistant"
            messages.append(
                Message(
                    role=role,
                    content=msg.content,
                )
            )

        metadata = ConversationMetadata(
            split=raw_conv.attributes.get("split"),
            source_dataset="Amod/mental_health_counseling_conversations",
            original_id=raw_conv.id,
            context=None,
        )

        return Conversation(
            id=raw_conv.id,
            source="Amod/mental_health_counseling_conversations",
            metadata=metadata,
            messages=messages,
        )
