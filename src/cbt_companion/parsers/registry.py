"""Registry mapping dataset identifiers to parser classes."""

from __future__ import annotations

from cbt_companion.parsers.base import BaseParser
from cbt_companion.parsers.empathetic_parser import EmpatheticParser
from cbt_companion.parsers.esconv_parser import ESConvParser
from cbt_companion.parsers.mental_health_parser import MentalHealthParser


class ParserRegistry:
    """Registry class holding available dataset parsers."""

    _registry: dict[str, type[BaseParser]] = {
        "facebook/empathetic_dialogues": EmpatheticParser,
        "thu-coai/esconv": ESConvParser,
        "Amod/mental_health_counseling_conversations": MentalHealthParser,
    }

    @classmethod
    def get_parser_class(cls, dataset_id: str) -> type[BaseParser]:
        """Get the parser class mapped to the given dataset ID.

        Args:
            dataset_id: The Hugging Face dataset ID or key.

        Returns:
            The mapped BaseParser subclass.

        Raises:
            KeyError: If no parser is registered for the dataset ID.
        """
        if dataset_id not in cls._registry:
            normalized = dataset_id.lower()
            for key, val in cls._registry.items():
                if key.lower() == normalized or key.split("/")[-1].lower() == normalized:
                    return val
            raise KeyError(
                f"No parser registered for dataset ID '{dataset_id}'. "
                f"Available parsers: {list(cls._registry.keys())}"
            )
        return cls._registry[dataset_id]

    @classmethod
    def register(cls, dataset_id: str, parser_class: type[BaseParser]) -> None:
        """Register a new parser class for a dataset ID.

        Args:
            dataset_id: Dataset ID/key to map.
            parser_class: Subclass of BaseParser.
        """
        cls._registry[dataset_id] = parser_class
