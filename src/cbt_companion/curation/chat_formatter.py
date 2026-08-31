"""Conversion from canonical conversations to chat-format training records."""

from __future__ import annotations

from cbt_companion.models.conversation import Conversation


class ChatFormatter:
    """Renders conversations as role-tagged message lists for supervised tuning.

    The output deliberately stops at a list of role/content dictionaries rather
    than a single pre-rendered string. Applying the model's chat template is left
    to the training notebook, so the prompt format is produced by the same
    tokenizer that serves it at inference and cannot drift out of sync.

    Attributes:
        system_prompt: Optional system turn prepended to every conversation.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt.strip() if system_prompt else None

    def format(self, conversation: Conversation) -> dict[str, list[dict[str, str]]]:
        """Render one conversation as a training record.

        Args:
            conversation: Curated conversation to render.

        Returns:
            A dictionary with a single "messages" key holding the role-tagged
            turns, optionally led by a system turn.
        """
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.extend(
            {"role": message.role, "content": message.content}
            for message in conversation.messages
        )
        return {"messages": messages}
