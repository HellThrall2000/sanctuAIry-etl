"""Text repair for corpus artifacts that would otherwise be learned verbatim."""

from __future__ import annotations

import html
import re

from cbt_companion.models.conversation import Conversation, Message
from cbt_companion.models.curation_report import CleaningStats

# EmpatheticDialogues encodes punctuation as underscore-wrapped placeholders.
_PLACEHOLDERS: dict[str, str] = {
    "_comma_": ",",
    "_period_": ".",
    "_quote_": '"',
}

# Sentence punctuation immediately followed by a capitalised word, produced by
# concatenating fields without a separator (e.g. "available.It sounds like").
_MISSING_SPACE = re.compile(r"(?<=[a-z])([.!?])(?=[A-Z])")

_URL = re.compile(r"https?://\S+|www\.\S+")

_REPEATED_WHITESPACE = re.compile(r"[^\S\n]{2,}")
_REPEATED_NEWLINES = re.compile(r"\n{3,}")


class TextCleaner:
    """Repairs known corpus artifacts in message content.

    The pipeline's normalizers copy source text verbatim, so artifacts from the
    upstream datasets reach the training set unchanged. A model fine-tuned on
    that text reproduces the artifacts in its own output, which is why these
    repairs run before any quality or sampling decision.

    Attributes:
        strip_urls: Whether bare URLs are removed from message content.
        stats: Mutable counters accumulated across calls.
    """

    def __init__(self, *, strip_urls: bool = True) -> None:
        self.strip_urls = strip_urls
        self.stats = CleaningStats()

    def clean_text(self, text: str) -> str:
        """Apply every repair rule to a single string.

        Args:
            text: Raw message content.

        Returns:
            The repaired string, stripped of leading and trailing whitespace.
        """
        cleaned = text

        for placeholder, replacement in _PLACEHOLDERS.items():
            if placeholder in cleaned:
                count = cleaned.count(placeholder)
                cleaned = cleaned.replace(placeholder, replacement)
                self._record(placeholder, count)

        unescaped = html.unescape(cleaned)
        if unescaped != cleaned:
            self._record("html_entity", 1)
            cleaned = unescaped

        repaired, n = _MISSING_SPACE.subn(r"\1 ", cleaned)
        if n:
            self._record("missing_space", n)
            cleaned = repaired

        if self.strip_urls:
            stripped, n = _URL.subn("", cleaned)
            if n:
                self._record("url", n)
                cleaned = stripped

        cleaned = _REPEATED_WHITESPACE.sub(" ", cleaned)
        cleaned = _REPEATED_NEWLINES.sub("\n\n", cleaned)
        return cleaned.strip()

    def clean_conversation(self, conversation: Conversation) -> Conversation:
        """Return a copy of the conversation with every message cleaned.

        Messages whose content becomes empty after cleaning are dropped; callers
        are responsible for discarding conversations left structurally invalid.

        Args:
            conversation: Canonical conversation to repair.

        Returns:
            A new Conversation carrying the repaired messages.
        """
        messages: list[Message] = []
        for message in conversation.messages:
            self.stats.messages_seen += 1
            cleaned = self.clean_text(message.content)
            if cleaned != message.content:
                self.stats.messages_changed += 1
            if not cleaned:
                continue
            messages.append(Message(role=message.role, content=cleaned))

        return conversation.model_copy(update={"messages": messages})

    def _record(self, rule: str, count: int) -> None:
        """Increment the replacement counter for a rule."""
        self.stats.replacements[rule] = self.stats.replacements.get(rule, 0) + count
