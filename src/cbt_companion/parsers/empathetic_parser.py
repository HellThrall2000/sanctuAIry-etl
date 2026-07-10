"""Parser for the EmpatheticDialogues dataset."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from cbt_companion.models.raw_conversation import (
    ParseReport,
    ParserInfo,
    RawConversation,
    RawMessage,
)
from cbt_companion.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class EmpatheticParser(BaseParser):
    """Parser for the facebook/empathetic_dialogues dataset."""

    def parse(self, dataset_split: Any, split_name: str) -> ParseReport:
        """Parse the EmpatheticDialogues dataset split.

        Args:
            dataset_split: The Hugging Face dataset split to parse.
            split_name: The name of the split (e.g. "train", "validation", "test").

        Returns:
            A ParseReport containing the parsed RawConversations and statistics.

        Raises:
            ValueError: If dataset-level unrecoverable checks fail.
        """
        logger.info("Parsing split '%s' of EmpatheticDialogues...", split_name)

        # Check for unrecoverable dataset-level failures: missing required columns
        required_cols = {"conv_id", "utterance_idx", "speaker_idx", "utterance"}
        if hasattr(dataset_split, "column_names"):
            cols = set(dataset_split.column_names)
        elif len(dataset_split) > 0 and hasattr(dataset_split[0], "keys"):
            cols = set(dataset_split[0].keys())
        else:
            cols = set()

        if len(dataset_split) > 0 and not required_cols.issubset(cols):
            missing = required_cols - cols
            raise ValueError(
                f"Dataset schema is invalid. Missing required column(s): {missing}"
            )

        warnings = []
        errors = []
        skipped_count = 0

        # 1. Group rows by conv_id
        grouped_rows = defaultdict(list)
        for i, row in enumerate(dataset_split):
            conv_id = row.get("conv_id")
            if conv_id is None or (isinstance(conv_id, str) and not conv_id.strip()):
                skipped_count += 1
                warning_msg = f"Skipping row {i}: missing a valid 'conv_id'."
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                continue
            grouped_rows[conv_id].append(row)

        conversations = []

        # 2. Sort and process each group
        for conv_id, rows in grouped_rows.items():
            try:
                # Verify and sort each group by utterance_idx
                for row in rows:
                    if "utterance_idx" not in row:
                        raise ValueError(f"Conversation '{conv_id}' is missing 'utterance_idx'.")
                rows.sort(key=lambda x: x["utterance_idx"])

                prev_idx = None
                raw_messages = []

                # Extract conversation-level attributes from the first row of the group
                first_row = rows[0]
                context = first_row.get("context")
                prompt = first_row.get("prompt")

                for row in rows:
                    idx = row.get("utterance_idx")
                    if idx is None:
                        raise ValueError(
                            f"Conversation '{conv_id}' has a message with missing 'utterance_idx'."
                        )
                    if prev_idx is not None and idx <= prev_idx:
                        raise ValueError(
                            f"Conversation '{conv_id}' has out-of-order indices: "
                            f"prev={prev_idx}, current={idx}."
                        )
                    prev_idx = idx

                    speaker_id = row.get("speaker_idx")
                    is_invalid_str = isinstance(speaker_id, str) and not speaker_id.strip()
                    if speaker_id is None or is_invalid_str:
                        raise ValueError(
                            f"Conversation '{conv_id}' has missing/empty speaker_idx."
                        )

                    # Validate message content (must not be empty)
                    content = row.get("utterance")
                    if content is None or not str(content):
                        raise ValueError(
                            f"Conversation '{conv_id}' at index {idx} has an empty 'utterance'."
                        )

                    # Build RawMessage
                    raw_messages.append(
                        RawMessage(
                            speaker_id=speaker_id,
                            content=str(content),
                            attributes={
                                "selfeval": row.get("selfeval"),
                                "tags": row.get("tags"),
                                "utterance_idx": idx,
                            },
                        )
                    )

                # Build RawConversation
                conversations.append(
                    RawConversation(
                        id=conv_id,
                        source="facebook/empathetic_dialogues",
                        messages=raw_messages,
                        attributes={
                            "context": context,
                            "prompt": prompt,
                            "split": split_name,
                        },
                    )
                )

            except ValueError as e:
                skipped_count += 1
                warning_msg = f"Skipping conversation '{conv_id}': {e}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)

        parser_info = ParserInfo(
            parser_name="EmpatheticParser",
            dataset_id="facebook/empathetic_dialogues",
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
