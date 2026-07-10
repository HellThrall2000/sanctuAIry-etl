"""Dataset registry — configuration-driven catalogue of all dataset sources."""

from __future__ import annotations

from collections.abc import Iterator, KeysView
from pathlib import Path
from typing import Self

import yaml

from cbt_companion.models.dataset import DatasetConfig


class DatasetRegistry:
    """Read-only registry of :class:`DatasetConfig` objects loaded from YAML.

    The registry is the single source of truth for which datasets the pipeline
    knows about.  It is deliberately *inert*: it stores metadata only and does
    not download, authenticate, or otherwise touch the referenced data.

    Usage::

        registry = DatasetRegistry.from_yaml(Path("configs/datasets.yaml"))

        entry = registry["esconv"]
        print(entry.name)  # "ESConv"

        for entry in registry:
            print(entry.key, entry.source)
    """

    # ── construction ─────────────────────────────────────────

    def __init__(self, entries: dict[str, DatasetConfig]) -> None:
        self._entries: dict[str, DatasetConfig] = dict(entries)

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Build a registry from a YAML configuration file.

        The file must contain a top-level ``datasets`` list.  Each item in the
        list is a mapping with the fields defined by :class:`DatasetConfig`.
        See ``configs/datasets.yaml`` for the canonical format.

        Args:
            path: Absolute or relative path to the YAML config file.

        Returns:
            A fully populated :class:`DatasetRegistry`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If the config is missing the ``datasets`` key.
            ValueError: If duplicate dataset keys are detected or an
                individual entry fails Pydantic validation.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict) or "datasets" not in raw:
            raise KeyError(f"No 'datasets' key found in {path}")

        dataset_list: list[dict] = raw["datasets"]
        if not isinstance(dataset_list, list):
            raise TypeError(
                f"'datasets' must be a list in {path}, got {type(dataset_list).__name__}"
            )

        entries: dict[str, DatasetConfig] = {}
        for item in dataset_list:
            config = DatasetConfig.model_validate(item)

            if config.key in entries:
                raise ValueError(
                    f"Duplicate dataset key '{config.key}' found in {path}. "
                    f"Each dataset must have a unique key."
                )

            entries[config.key] = config

        return cls(entries)

    # ── public interface ─────────────────────────────────────

    def get(self, key: str) -> DatasetConfig | None:
        """Return the entry for *key*, or ``None`` if it is not registered."""
        return self._entries.get(key)

    def keys(self) -> KeysView[str]:
        """Return a view of all registered dataset keys."""
        return self._entries.keys()

    def list_all(self) -> list[DatasetConfig]:
        """Return all registered dataset configs as a list."""
        return list(self._entries.values())

    # ── dunder protocol ──────────────────────────────────────

    def __getitem__(self, key: str) -> DatasetConfig:
        try:
            return self._entries[key]
        except KeyError:
            registered = ", ".join(sorted(self._entries)) or "(none)"
            raise KeyError(
                f"Dataset '{key}' is not registered. "
                f"Available keys: {registered}"
            ) from None

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[DatasetConfig]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._entries))
        return f"DatasetRegistry([{keys}])"
