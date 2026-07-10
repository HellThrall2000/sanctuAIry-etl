"""Parser for the ESConv dataset."""

from __future__ import annotations

import json
import logging
from typing import Any

from cbt_companion.models.raw_conversation import (
    ParseReport,
    ParserInfo,
    RawConversation,
    RawMessage,
)
from cbt_companion.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class ESConvParser(BaseParser):
    """Parser for the thu-coai/esconv dataset."""

    def parse(self, dataset_split: Any, split_name: str) -> ParseReport:
        """Parse the ESConv dataset split.

        Args:
            dataset_split: The Hugging Face dataset split to parse.
            split_name: The name of the split (e.g. "train", "validation", "test").

        Returns:
            A ParseReport containing the parsed RawConversations and statistics.

        Raises:
            ValueError: If dataset-level unrecoverable checks fail.
        """
        logger.info("Parsing split '%s' of ESConv...", split_name)

        # Check for unrecoverable dataset-level failures: missing required columns
        if hasattr(dataset_split, "column_names"):
            cols = set(dataset_split.column_names)
        elif len(dataset_split) > 0 and hasattr(dataset_split[0], "keys"):
            cols = set(dataset_split[0].keys())
        else:
            cols = set()

        if len(dataset_split) > 0 and "text" not in cols:
            raise ValueError("Dataset schema is invalid. Missing required column: 'text'")

        conversations = []
        warnings = []
        errors = []
        skipped_count = 0

        for i, row in enumerate(dataset_split):
            conv_id = f"esconv_{split_name}_{i}"
            try:
                raw_text = row.get("text")
                if raw_text is None:
                    raise ValueError("Row is missing the 'text' field.")

                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Row contains invalid JSON: {e}") from e

                # Extract dialog
                dialog = data.get("dialog")
                if dialog is None or not isinstance(dialog, list):
                    raise ValueError("Row is missing a valid 'dialog' list.")

                raw_messages = []
                for j, turn in enumerate(dialog):
                    if not isinstance(turn, dict):
                        raise ValueError(f"Turn at index {j} is not a dictionary.")

                    content = turn.get("text")
                    if content is None or not str(content):
                        raise ValueError(f"Turn {j} has empty/missing text.")

                    speaker_id = turn.get("speaker")
                    is_invalid_str = isinstance(speaker_id, str) and not speaker_id.strip()
                    if speaker_id is None or is_invalid_str:
                        raise ValueError(f"Turn {j} has a missing/empty speaker.")

                    # Message-level attributes (exclude text and speaker)
                    msg_attributes = {k: v for k, v in turn.items() if k not in ("text", "speaker")}

                    raw_messages.append(
                        RawMessage(
                            speaker_id=speaker_id,
                            content=str(content),
                            attributes=msg_attributes,
                        )
                    )

                # Conversation-level attributes (exclude dialog)
                conv_attributes = {k: v for k, v in data.items() if k != "dialog"}
                conv_attributes["split"] = split_name

                conversations.append(
                    RawConversation(
                        id=conv_id,
                        source="thu-coai/esconv",
                        messages=raw_messages,
                        attributes=conv_attributes,
                    )
                )

            except ValueError as e:
                skipped_count += 1
                warning_msg = f"Skipping conversation '{conv_id}': {e}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)

        parser_info = ParserInfo(
            parser_name="ESConvParser",
            dataset_id="thu-coai/esconv",
        )

        report = ParseReport.from_conversations(
            conversations=conversations,
            parser_info=parser_info,
            split=split_name,
            skipped_conversations=skipped_count,
            warnings=warnings,
            errors=errors,
        )

        logger.info(
            "Successfully parsed %d conversations from split '%s' (skipped %d).",
            report.num_conversations,
            split_name,
            report.skipped_conversations,
        )
        return report
