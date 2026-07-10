"""Registry mapping dataset identifiers to normalizer classes."""

from __future__ import annotations

from cbt_companion.normalizers.base import BaseNormalizer
from cbt_companion.normalizers.empathetic_normalizer import EmpatheticNormalizer
from cbt_companion.normalizers.esconv_normalizer import ESConvNormalizer
from cbt_companion.normalizers.mental_health_normalizer import MentalHealthNormalizer


class NormalizerRegistry:
    """Registry class holding available dataset normalizers."""

    _registry: dict[str, type[BaseNormalizer]] = {
        "facebook/empathetic_dialogues": EmpatheticNormalizer,
        "thu-coai/esconv": ESConvNormalizer,
        "Amod/mental_health_counseling_conversations": MentalHealthNormalizer,
    }

    @classmethod
    def get_normalizer_class(cls, dataset_id: str) -> type[BaseNormalizer]:
        """Get the normalizer class mapped to the given dataset ID.

        Args:
            dataset_id: The Hugging Face dataset ID or key.

        Returns:
            The mapped BaseNormalizer subclass.

        Raises:
            KeyError: If no normalizer is registered for the dataset ID.
        """
        if dataset_id not in cls._registry:
            normalized = dataset_id.lower()
            for key, val in cls._registry.items():
                if key.lower() == normalized or key.split("/")[-1].lower() == normalized:
                    return val
            raise KeyError(
                f"No normalizer registered for dataset ID '{dataset_id}'. "
                f"Available normalizers: {list(cls._registry.keys())}"
            )
        return cls._registry[dataset_id]

    @classmethod
    def register(cls, dataset_id: str, normalizer_class: type[BaseNormalizer]) -> None:
        """Register a new normalizer class for a dataset ID.

        Args:
            dataset_id: Dataset ID/key to map.
            normalizer_class: Subclass of BaseNormalizer.
        """
        cls._registry[dataset_id] = normalizer_class
