"""Tool for computing dataset statistics and generating review samples."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from cbt_companion.models.statistics_report import StatisticsReport

if TYPE_CHECKING:
    from cbt_companion.models.conversation import Conversation
    from cbt_companion.models.merge_report import MergeReport


class StatisticsTool:
    """Computes dataset statistics and generates review samples in-memory."""

    @staticmethod
    def compute_statistics(merge_report: MergeReport) -> StatisticsReport:
        """Calculate statistics from a MergeReport.

        Args:
            merge_report: MergeReport containing the list of merged conversations.

        Returns:
            A StatisticsReport containing computed metrics.
        """
        conversations = merge_report.merged_conversations
        total_convs = len(conversations)

        if total_convs == 0:
            return StatisticsReport(
                total_conversations=0,
                conversations_per_dataset={},
                percentage_contribution_per_dataset={},
                average_turns=0.0,
                median_turns=0.0,
                minimum_turns=0,
                maximum_turns=0,
                average_user_message_length=0.0,
                average_assistant_message_length=0.0,
                maximum_message_length=0,
                context_distribution={},
            )

        # 1. Conversations per dataset & contribution percentage
        convs_per_dataset: dict[str, int] = {}
        for conv in conversations:
            convs_per_dataset[conv.source] = convs_per_dataset.get(conv.source, 0) + 1

        percent_contribution: dict[str, float] = {}
        for source, count in convs_per_dataset.items():
            percent_contribution[source] = round((count / total_convs) * 100.0, 2)

        # 2. Turn count metrics
        turn_counts = [len(conv.messages) for conv in conversations]
        average_turns = round(sum(turn_counts) / total_convs, 2)

        # Median turns
        sorted_turns = sorted(turn_counts)
        n = len(sorted_turns)
        if n % 2 == 1:
            median_turns = float(sorted_turns[n // 2])
        else:
            median_turns = (sorted_turns[n // 2 - 1] + sorted_turns[n // 2]) / 2.0

        minimum_turns = min(turn_counts)
        maximum_turns = max(turn_counts)

        # 3. Message length metrics (character lengths)
        user_lens: list[int] = []
        assistant_lens: list[int] = []
        all_lens: list[int] = []

        for conv in conversations:
            for msg in conv.messages:
                length = len(msg.content)
                all_lens.append(length)
                if msg.role == "user":
                    user_lens.append(length)
                elif msg.role == "assistant":
                    assistant_lens.append(length)

        avg_user_len = round(sum(user_lens) / len(user_lens), 2) if user_lens else 0.0
        avg_assistant_len = (
            round(sum(assistant_lens) / len(assistant_lens), 2)
            if assistant_lens
            else 0.0
        )
        max_msg_len = max(all_lens) if all_lens else 0

        # 4. Context distribution (top contexts and counts)
        contexts: dict[str, int] = {}
        for conv in conversations:
            ctx = conv.metadata.context
            ctx_name = str(ctx) if ctx is not None else "None"
            contexts[ctx_name] = contexts.get(ctx_name, 0) + 1

        # Sort context distribution by count descending
        sorted_contexts = dict(sorted(contexts.items(), key=lambda item: item[1], reverse=True))

        return StatisticsReport(
            total_conversations=total_convs,
            conversations_per_dataset=convs_per_dataset,
            percentage_contribution_per_dataset=percent_contribution,
            average_turns=average_turns,
            median_turns=median_turns,
            minimum_turns=minimum_turns,
            maximum_turns=maximum_turns,
            average_user_message_length=avg_user_len,
            average_assistant_message_length=avg_assistant_len,
            maximum_message_length=max_msg_len,
            context_distribution=sorted_contexts,
        )

    @staticmethod
    def generate_sample(
        merge_report: MergeReport,
        sample_size: int = 100,
        seed: int = 42,
    ) -> list[Conversation]:
        """Generate a random sample of conversations from a MergeReport.

        Uses a deterministic random seed to select conversations, preserving
        their original order within the dataset.

        Args:
            merge_report: MergeReport containing all merged conversations.
            sample_size: Number of conversations to sample.
            seed: Random seed for deterministic selection.

        Returns:
            A list of sampled Conversation objects.
        """
        conversations = merge_report.merged_conversations
        total_convs = len(conversations)

        if total_convs <= sample_size:
            return list(conversations)

        rng = random.Random(seed)
        indices = list(range(total_convs))
        sampled_indices = rng.sample(indices, sample_size)
        sampled_indices.sort()  # Preserve original dataset order

        return [conversations[idx] for idx in sampled_indices]
