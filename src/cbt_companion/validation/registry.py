"""Registry mapping validator identifiers to validator classes."""

from __future__ import annotations

from cbt_companion.validation.base import BaseValidator
from cbt_companion.validation.conversation_validator import ConversationValidator


class ValidatorRegistry:
    """Registry class holding available dataset validators."""

    _registry: dict[str, type[BaseValidator]] = {
        "conversation": ConversationValidator,
    }

    @classmethod
    def get_validator_class(cls, validator_id: str) -> type[BaseValidator]:
        """Get the validator class mapped to the given identifier.

        Args:
            validator_id: The identifier for the validator.

        Returns:
            The mapped BaseValidator subclass.

        Raises:
            KeyError: If no validator is registered for the identifier.
        """
        if validator_id not in cls._registry:
            normalized = validator_id.lower()
            for key, val in cls._registry.items():
                if key.lower() == normalized:
                    return val
            raise KeyError(
                f"No validator registered for identifier '{validator_id}'. "
                f"Available validators: {list(cls._registry.keys())}"
            )
        return cls._registry[validator_id]

    @classmethod
    def register(cls, validator_id: str, validator_class: type[BaseValidator]) -> None:
        """Register a new validator class.

        Args:
            validator_id: Identifier to map.
            validator_class: Subclass of BaseValidator.
        """
        cls._registry[validator_id] = validator_class
