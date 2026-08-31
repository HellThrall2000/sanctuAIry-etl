"""Orchestration of the curation stage: clean, filter, rebalance, split, export."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from cbt_companion.curation.chat_formatter import ChatFormatter
from cbt_companion.curation.quality_filter import QualityFilter
from cbt_companion.curation.sampler import SourceSampler
from cbt_companion.curation.text_cleaner import TextCleaner
from cbt_companion.curation.trimmer import ConversationTrimmer
from cbt_companion.curation.turn_normalizer import TurnNormalizer
from cbt_companion.models.conversation import Conversation
from cbt_companion.models.curation_report import CurationReport

logger = logging.getLogger("cbt_companion.curation")


def _count_by_source(conversations: list[Conversation]) -> dict[str, int]:
    """Tally conversations per source identifier."""
    counts: dict[str, int] = {}
    for conversation in conversations:
        counts[conversation.source] = counts.get(conversation.source, 0) + 1
    return counts


class DatasetCurator:
    """Turns deduplicated conversations into train and validation JSONL files.

    Attributes:
        cleaner: Text-repair stage.
        turn_normalizer: Turn-alternation reshaping stage.
        trimmer: Trailing-turn truncation stage.
        quality_filter: Content-quality stage.
        sampler: Per-source rebalancing stage.
        formatter: Chat-record renderer.
        split_ratio: Fraction of conversations assigned to the training split.
        seed: Random seed for the train/validation shuffle.
    """

    def __init__(
        self,
        *,
        cleaner: TextCleaner,
        quality_filter: QualityFilter,
        trimmer: ConversationTrimmer | None = None,
        turn_normalizer: TurnNormalizer | None = None,
        sampler: SourceSampler,
        formatter: ChatFormatter,
        split_ratio: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.cleaner = cleaner
        self.turn_normalizer = turn_normalizer or TurnNormalizer()
        self.trimmer = trimmer or ConversationTrimmer()
        self.quality_filter = quality_filter
        self.sampler = sampler
        self.formatter = formatter
        self.split_ratio = split_ratio
        self.seed = seed

    def curate(
        self,
        conversations: list[Conversation],
        train_path: Path,
        validation_path: Path,
    ) -> CurationReport:
        """Run every curation stage and write both splits to disk.

        Args:
            conversations: Deduplicated canonical conversations.
            train_path: Destination for the training split.
            validation_path: Destination for the validation split.

        Returns:
            A CurationReport describing what each stage changed or removed.
        """
        report = CurationReport(seed=self.seed)
        report.counts_before = _count_by_source(conversations)

        logger.info("Cleaning text artifacts across %d conversations...", len(conversations))
        cleaned = [self.cleaner.clean_conversation(c) for c in conversations]

        logger.info("Normalizing turn alternation...")
        shaped = [self.turn_normalizer.normalize(c) for c in cleaned]

        logger.info("Trimming trailing turns with no assistant response...")
        trimmed = [self.trimmer.trim(c) for c in shaped]

        logger.info("Applying content-quality rules...")
        kept = self.quality_filter.filter(trimmed)
        report.counts_after_quality = _count_by_source(kept)
        logger.info("Quality filter retained %d of %d conversations.", len(kept), len(trimmed))

        logger.info("Rebalancing sources against configured caps...")
        sampled = self.sampler.sample(kept)
        report.counts_after_sampling = _count_by_source(sampled)
        logger.info("Rebalancing retained %d conversations.", len(sampled))

        rng = random.Random(self.seed)
        rng.shuffle(sampled)
        split_index = int(len(sampled) * self.split_ratio)
        train, validation = sampled[:split_index], sampled[split_index:]

        self._write(train, train_path)
        self._write(validation, validation_path)

        report.cleaning = self.cleaner.stats
        report.quality = self.quality_filter.stats
        report.num_train = len(train)
        report.num_validation = len(validation)
        return report

    def _write(self, conversations: list[Conversation], path: Path) -> None:
        """Write conversations to a JSONL file as chat-format records."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for conversation in conversations:
                record = self.formatter.format(conversation)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Wrote %d conversations to %s", len(conversations), path)
