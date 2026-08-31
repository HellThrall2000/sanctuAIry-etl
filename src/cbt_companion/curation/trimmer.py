"""Trimming of turns that cannot contribute a useful training signal."""

from __future__ import annotations

from cbt_companion.models.conversation import Conversation


class ConversationTrimmer:
    """Truncates a conversation at its last substantive assistant turn.

    Supervised fine-tuning computes loss over assistant turns, so a conversation
    ending on a user turn contributes a prompt with no target. Transcripts also
    routinely close on a pleasantry ("Good luck!"), which makes a poor final
    generation target and would otherwise cost the whole conversation.

    Truncating recovers the usable prefix instead of discarding the conversation,
    which matters most for the long multi-turn counselling dialogues that are the
    scarcest source in the corpus.

    Attributes:
        min_final_assistant_chars: Length an assistant turn must reach to be kept
            as the final turn.
    """

    def __init__(self, *, min_final_assistant_chars: int = 40) -> None:
        self.min_final_assistant_chars = min_final_assistant_chars

    def trim(self, conversation: Conversation) -> Conversation:
        """Truncate the conversation at its last substantive assistant turn.

        Args:
            conversation: Cleaned conversation to trim.

        Returns:
            A conversation ending on an assistant turn of at least
            min_final_assistant_chars, or one with no messages when no such turn
            exists.
        """
        messages = conversation.messages
        cut = -1
        for index, message in enumerate(messages):
            if (
                message.role == "assistant"
                and len(message.content) >= self.min_final_assistant_chars
            ):
                cut = index

        if cut == -1:
            return conversation.model_copy(update={"messages": []})
        if cut == len(messages) - 1:
            return conversation

        return conversation.model_copy(update={"messages": messages[: cut + 1]})
