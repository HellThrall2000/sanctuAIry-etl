"""Parser for the Mental Health Counseling Conversations dataset."""

from __future__ import annotations

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


class MentalHealthParser(BaseParser):
    """Parser for the Amod/mental_health_counseling_conversations dataset."""

    def parse(self, dataset_split: Any, split_name: str) -> ParseReport:
        """Parse the Mental Health dataset split.

        Args:
            dataset_split: The Hugging Face dataset split to parse.
            split_name: The name of the split (e.g. "train").

        Returns:
            A ParseReport containing the parsed RawConversations and statistics.

        Raises:
            ValueError: If dataset-level unrecoverable checks fail.
        """
        logger.info("Parsing split '%s' of Mental Health...", split_name)

        # Check for unrecoverable dataset-level failures: missing required columns
        if hasattr(dataset_split, "column_names"):
            cols = set(dataset_split.column_names)
        elif len(dataset_split) > 0 and hasattr(dataset_split[0], "keys"):
            cols = set(dataset_split[0].keys())
        else:
            cols = set()

        required_cols = {"Context", "Response"}
        if len(dataset_split) > 0 and not required_cols.issubset(cols):
            missing = required_cols - cols
            raise ValueError(
                f"Dataset schema is invalid. Missing required column(s): {missing}"
            )

        conversations = []
        warnings = []
        errors = []
        skipped_count = 0

        for i, row in enumerate(dataset_split):
            conv_id = f"mental_health_{split_name}_{i}"
            try:
                context = row.get("Context")
                response = row.get("Response")

                if context is None or not str(context).strip():
                    raise ValueError("Row has empty or missing 'Context'.")

                if response is None or not str(response).strip():
                    raise ValueError("Row has empty or missing 'Response'.")

                raw_messages = [
                    RawMessage(
                        speaker_id="Context",
                        content=str(context),
                        attributes={},
                    ),
                    RawMessage(
                        speaker_id="Response",
                        content=str(response),
                        attributes={},
                    ),
                ]

                conversations.append(
                    RawConversation(
                        id=conv_id,
                        source="Amod/mental_health_counseling_conversations",
                        messages=raw_messages,
                        attributes={
                            "Context": context,
                            "Response": response,
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
            parser_name="MentalHealthParser",
            dataset_id="Amod/mental_health_counseling_conversations",
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
