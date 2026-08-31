"""Content-quality filtering for the training export."""

from __future__ import annotations

import difflib
import re

from cbt_companion.models.conversation import Conversation
from cbt_companion.models.curation_report import QualityStats

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def _normalize_for_compare(text: str) -> str:
    """Lowercase and strip punctuation so echoes differing only in style match."""
    return _NON_ALNUM.sub("", text.lower()).strip()


class QualityFilter:
    """Drops conversations that would teach undesirable response behaviour.

    The pipeline's existing ConversationFilter enforces schema validity only, so
    structurally valid but pedagogically harmful conversations survive to the
    training set.

    The length rules are deliberately not a per-turn veto. Genuine counselling
    dialogue contains short backchannels ("I see."), and rejecting a whole
    conversation for one of them discards exactly the long multi-turn transcripts
    the corpus has fewest of. Instead the final assistant turn — the primary
    generation target — must be substantive, and short turns must not dominate
    the conversation as a whole.

    Attributes:
        min_assistant_chars: Minimum length of the final assistant turn.
        max_short_assistant_ratio: Maximum fraction of assistant turns allowed to
            fall below min_assistant_chars.
        echo_similarity_threshold: Similarity above which an assistant turn is
            treated as an echo of the preceding user turn.
        min_messages: Minimum number of messages after cleaning and trimming.
        stats: Mutable counters accumulated across calls.
    """

    def __init__(
        self,
        *,
        min_assistant_chars: int = 40,
        max_short_assistant_ratio: float = 0.5,
        echo_similarity_threshold: float = 0.8,
        min_messages: int = 2,
    ) -> None:
        self.min_assistant_chars = min_assistant_chars
        self.max_short_assistant_ratio = max_short_assistant_ratio
        self.echo_similarity_threshold = echo_similarity_threshold
        self.min_messages = min_messages
        self.stats = QualityStats()

    def evaluate(self, conversation: Conversation) -> str | None:
        """Assess a single conversation against every quality rule.

        Args:
            conversation: Cleaned and trimmed conversation to assess.

        Returns:
            The name of the first rule the conversation violates, or None when
            the conversation passes.
        """
        messages = conversation.messages

        if len(messages) < self.min_messages:
            return "too_few_messages"

        assistant_turns = [m for m in messages if m.role == "assistant"]
        if not assistant_turns:
            return "no_assistant_turn"

        if len(assistant_turns[-1].content) < self.min_assistant_chars:
            return "final_response_too_short"

        short = sum(1 for m in assistant_turns if len(m.content) < self.min_assistant_chars)
        if short / len(assistant_turns) > self.max_short_assistant_ratio:
            return "mostly_short_responses"

        if self._has_echo(messages):
            return "assistant_echoes_user"

        return None

    def filter(self, conversations: list[Conversation]) -> list[Conversation]:
        """Retain only conversations that pass every quality rule.

        Args:
            conversations: Cleaned and trimmed conversations to filter.

        Returns:
            The retained conversations, in input order.
        """
        kept: list[Conversation] = []
        for conversation in conversations:
            self.stats.num_seen += 1
            reason = self.evaluate(conversation)
            if reason is None:
                kept.append(conversation)
            else:
                self.stats.drops_by_reason[reason] = (
                    self.stats.drops_by_reason.get(reason, 0) + 1
                )

        self.stats.num_kept = len(kept)
        return kept

    def _has_echo(self, messages: list) -> bool:
        """Report whether any assistant turn restates the user turn before it."""
        for index, message in enumerate(messages):
            if message.role != "assistant" or index == 0:
                continue
            previous = messages[index - 1]
            if previous.role != "user":
                continue
            left = _normalize_for_compare(previous.content)
            right = _normalize_for_compare(message.content)
            if not left or not right:
                continue
            if difflib.SequenceMatcher(None, left, right).ratio() >= (
                self.echo_similarity_threshold
            ):
                return True
        return False
