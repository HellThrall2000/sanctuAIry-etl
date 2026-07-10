"""Normalizer for the EmpatheticDialogues dataset."""

from __future__ import annotations

from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message
from cbt_companion.models.raw_conversation import RawConversation
from cbt_companion.normalizers.base import BaseNormalizer


class EmpatheticNormalizer(BaseNormalizer):
    """Normalizes EmpatheticDialogues raw conversations."""

    def normalize(self, raw_conv: RawConversation) -> Conversation:
        """Convert EmpatheticDialogues RawConversation to canonical Conversation."""
        messages = []
        if not raw_conv.messages:
            raise ValueError(f"Raw conversation '{raw_conv.id}' has no messages.")

        # Alternating speaker logic
        current_role = "user"
        prev_speaker = raw_conv.messages[0].speaker_id

        for msg in raw_conv.messages:
            if msg.speaker_id != prev_speaker:
                current_role = "assistant" if current_role == "user" else "user"
                prev_speaker = msg.speaker_id

            messages.append(
                Message(
                    role=current_role,
                    content=msg.content,
                )
            )

        metadata = ConversationMetadata(
            split=raw_conv.attributes.get("split"),
            source_dataset="facebook/empathetic_dialogues",
            original_id=raw_conv.id,
            context=raw_conv.attributes.get("context"),
        )

        return Conversation(
            id=raw_conv.id,
            source="facebook/empathetic_dialogues",
            metadata=metadata,
            messages=messages,
        )
