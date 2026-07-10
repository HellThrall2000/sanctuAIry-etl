"""Abstract base class for raw conversation normalizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cbt_companion.models.conversation import Conversation
from cbt_companion.models.raw_conversation import RawConversation


class BaseNormalizer(ABC):
    """Abstract base class for converting RawConversation to canonical Conversation."""

    @abstractmethod
    def normalize(self, raw_conv: RawConversation) -> Conversation:
        """Convert a raw conversation into the canonical Conversation model.

        Args:
            raw_conv: The raw conversation input.

        Returns:
            The normalized canonical Conversation output.
        """
        pass
