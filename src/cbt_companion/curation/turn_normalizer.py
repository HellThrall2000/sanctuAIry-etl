"""Structural normalization of turn sequences for chat-template rendering."""

from __future__ import annotations

from cbt_companion.models.conversation import Conversation, Message


class TurnNormalizer:
    """Reshapes turn sequences into the strict alternation chat templates expect.

    The dataset normalizers assign a role per source utterance, so a speaker
    taking two consecutive turns yields two consecutive same-role messages, and
    transcripts where the counsellor speaks first begin on an assistant turn.
    Gemma's chat template — like most — requires the first non-system turn to be
    a user turn and roles to alternate strictly thereafter, and raises on input
    that does not. Reshaping here means the training text and the text the model
    is served at inference are produced by the same template without error.

    Attributes:
        separator: String joining the content of merged consecutive turns.
    """

    def __init__(self, *, separator: str = "\n\n") -> None:
        self.separator = separator

    def normalize(self, conversation: Conversation) -> Conversation:
        """Merge consecutive same-role turns and drop any leading assistant turn.

        Args:
            conversation: Cleaned conversation to reshape.

        Returns:
            A conversation whose turns strictly alternate starting from a user
            turn, or one with no messages when nothing usable remains.
        """
        merged: list[Message] = []
        for message in conversation.messages:
            if merged and merged[-1].role == message.role:
                combined = f"{merged[-1].content}{self.separator}{message.content}"
                merged[-1] = Message(role=message.role, content=combined)
            else:
                merged.append(message)

        # A transcript opening on the counsellor has no prompt for its first
        # response, so that turn is dropped rather than merged forward.
        if merged and merged[0].role == "assistant":
            merged = merged[1:]

        return conversation.model_copy(update={"messages": merged})
