"""Deterministic per-source downsampling."""

from __future__ import annotations

import random

from cbt_companion.models.conversation import Conversation


class SourceSampler:
    """Caps the number of conversations contributed by each source.

    The merged corpus is dominated by a single source, which biases the training
    signal toward that source's response style. Applying a per-source cap
    rebalances the mix without discarding the smaller, higher-signal sources.

    Attributes:
        caps: Mapping of source identifier to maximum conversation count. A
            missing key or a None value means the source is left uncapped.
        seed: Random seed guaranteeing reproducible selection.
    """

    def __init__(self, caps: dict[str, int | None], *, seed: int = 42) -> None:
        self.caps = caps
        self.seed = seed

    def sample(self, conversations: list[Conversation]) -> list[Conversation]:
        """Apply per-source caps, preferring the longest conversations.

        Within a capped source, conversations are ranked by total assistant
        character count so the retained subset carries the most substantive
        responses rather than an arbitrary slice.

        Args:
            conversations: Conversations to downsample.

        Returns:
            The retained conversations, deterministically shuffled.
        """
        by_source: dict[str, list[Conversation]] = {}
        for conversation in conversations:
            by_source.setdefault(conversation.source, []).append(conversation)

        kept: list[Conversation] = []
        for source, group in by_source.items():
            cap = self.caps.get(source)
            if cap is None or cap >= len(group):
                kept.extend(group)
                continue

            ranked = sorted(group, key=self._assistant_chars, reverse=True)
            kept.extend(ranked[:cap])

        rng = random.Random(self.seed)
        rng.shuffle(kept)
        return kept

    @staticmethod
    def _assistant_chars(conversation: Conversation) -> tuple[int, str]:
        """Total assistant characters, with the ID breaking ties deterministically."""
        total = sum(len(m.content) for m in conversation.messages if m.role == "assistant")
        return total, conversation.id
