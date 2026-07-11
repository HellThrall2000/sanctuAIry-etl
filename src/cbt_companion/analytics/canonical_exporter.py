"""Canonical dataset exporter for writing final merged conversations to JSONL."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cbt_companion.models.conversation import Conversation


class CanonicalDatasetExporter:
    """Writes a sequence of canonical Conversation objects to a JSONL file.

    Each line contains a single Conversation serialised as a JSON object,
    preserving the full Pydantic schema and all metadata.  Output is
    UTF-8 encoded with one JSON object per line (no wrapping array).
    """

    @staticmethod
    def export(
        conversations: Sequence[Conversation],
        output_path: str | Path,
    ) -> int:
        """Export canonical conversations to a JSONL file.

        Args:
            conversations: Final merged Conversation objects to export.
            output_path: File path for the JSONL output.

        Returns:
            The number of conversations written.
        """
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            for conv in conversations:
                fh.write(conv.model_dump_json() + "\n")

        return len(conversations)
