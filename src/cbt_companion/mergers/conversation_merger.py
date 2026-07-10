"""Merger implementation combining multiple Conversation sequences."""

from __future__ import annotations

from collections.abc import Sequence

from cbt_companion.mergers.base import BaseMerger, MergeError
from cbt_companion.models.conversation import Conversation
from cbt_companion.models.merge_report import MergeReport


class ConversationMerger(BaseMerger):
    """Combines multiple sequences of Conversation objects into a single sequence."""

    def merge(self, datasets: Sequence[Sequence[Conversation]]) -> MergeReport:
        """Merge multiple datasets of Conversation objects.

        Args:
            datasets: A sequence of sequences containing validated/filtered Conversations.

        Returns:
            A MergeReport detailing the merged output and contribution stats.

        Raises:
            MergeError: If a duplicate conversation ID is detected across datasets.
        """
        merged_conversations = []
        seen_ids = set()
        contribution_counts = {}
        dataset_order = []

        for i, dataset in enumerate(datasets):
            # Inspect first conversation to record dataset source name in order
            if len(dataset) > 0:
                first_conv = dataset[0]
                source_name = first_conv.source
                dataset_order.append(source_name)
            else:
                continue

            for conv in dataset:
                if conv.id in seen_ids:
                    raise MergeError(
                        f"Merge collision: Duplicate conversation ID '{conv.id}' "
                        f"detected in dataset segment {i} ({conv.source})."
                    )
                seen_ids.add(conv.id)
                merged_conversations.append(conv)

                # Track contribution count
                contribution_counts[conv.source] = contribution_counts.get(conv.source, 0) + 1

        return MergeReport(
            merged_conversations=merged_conversations,
            total_conversations=len(merged_conversations),
            contribution_counts=contribution_counts,
            dataset_order=dataset_order,
        )
