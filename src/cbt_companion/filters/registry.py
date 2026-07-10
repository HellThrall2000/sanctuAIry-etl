"""Registry mapping filter identifiers to filter classes."""

from __future__ import annotations

from cbt_companion.filters.base import BaseFilter
from cbt_companion.filters.conversation_filter import ConversationFilter


class FilterRegistry:
    """Registry class holding available dataset filters."""

    _registry: dict[str, type[BaseFilter]] = {
        "conversation": ConversationFilter,
    }

    @classmethod
    def get_filter_class(cls, filter_id: str) -> type[BaseFilter]:
        """Get the filter class mapped to the given identifier.

        Args:
            filter_id: The identifier for the filter.

        Returns:
            The mapped BaseFilter subclass.

        Raises:
            KeyError: If no filter is registered for the identifier.
        """
        if filter_id not in cls._registry:
            normalized = filter_id.lower()
            for key, val in cls._registry.items():
                if key.lower() == normalized:
                    return val
            raise KeyError(
                f"No filter registered for identifier '{filter_id}'. "
                f"Available filters: {list(cls._registry.keys())}"
            )
        return cls._registry[filter_id]

    @classmethod
    def register(cls, filter_id: str, filter_class: type[BaseFilter]) -> None:
        """Register a new filter class.

        Args:
            filter_id: Identifier to map.
            filter_class: Subclass of BaseFilter.
        """
        cls._registry[filter_id] = filter_class
