"""Exact conversation deduplicator stage for canonical datasets."""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cbt_companion.models.conversation import Conversation


class ConversationDeduplicator:
    """Performs deterministic exact deduplication of Conversation objects."""

    @staticmethod
    def normalize_content(content: str) -> str:
        """Applies harmless formatting normalization to message content text.

        - Normalize line endings (replace \\r\\n and \\r with \\n).
        - Convert to lowercase.
        - Collapse multiple whitespace characters into a single space.
        - Trim leading and trailing whitespace.
        """
        # Normalize line endings
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        # Convert to lowercase
        normalized = normalized.lower()
        # Collapse multiple whitespace characters into a single space and trim
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @classmethod
    def compute_hash(cls, messages: list[Any]) -> str:
        """Computes a SHA-256 hash for a normalized list of messages.

        Only serializes role and normalized content. Ignore all metadata.
        """
        serialized_messages = []
        for msg in messages:
            serialized_messages.append({
                "role": msg.role,
                "content": cls.normalize_content(msg.content),
            })

        # Deterministic serialization using sorted keys and compact separators
        serialized = json.dumps(
            serialized_messages,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def deduplicate(
        cls,
        conversations: list[Conversation],
        dataset_version: str = "1.0.0",
    ) -> tuple[list[Conversation], dict[str, Any], dict[str, Any], str]:
        """Deduplicates a list of conversations in-memory.

        Keep the first occurrence of a unique hash, discard subsequent ones.
        Preserves original order.

        Args:
            conversations: Input list of canonical Conversation objects.
            dataset_version: Version string for the manifest.

        Returns:
            A tuple of:
                - deduplicated_conversations: list of unique Conversations.
                - manifest: manifest dictionary.
                - duplicate_report_json: structured duplicate report dictionary.
                - duplicate_report_md: human-readable Markdown report string.
        """
        deduplicated: list[Conversation] = []
        seen_hashes: dict[str, Conversation] = {}  # sha256 -> first conversation instance
        removed_duplicates_info: list[dict[str, str]] = []

        for conv in conversations:
            sha256_hash = cls.compute_hash(conv.messages)
            if sha256_hash not in seen_hashes:
                seen_hashes[sha256_hash] = conv
                deduplicated.append(conv)
            else:
                kept_conv = seen_hashes[sha256_hash]
                removed_duplicates_info.append({
                    "kept_conversation_id": kept_conv.id,
                    "removed_conversation_id": conv.id,
                    "kept_source": kept_conv.source,
                    "removed_source": conv.source,
                    "sha256_hash": sha256_hash,
                })

        total_before = len(conversations)
        total_after = len(deduplicated)
        duplicates_removed = len(removed_duplicates_info)

        if total_before > 0:
            duplicate_percentage = round((duplicates_removed / total_before) * 100.0, 2)
        else:
            duplicate_percentage = 0.0

        # Build manifest
        created_at = datetime.datetime.now(datetime.UTC).isoformat()
        manifest = {
            "dataset_version": dataset_version,
            "created_at": created_at,
            "total_conversations_before": total_before,
            "total_conversations_after": total_after,
            "duplicates_removed": duplicates_removed,
            "duplicate_percentage": duplicate_percentage,
        }

        # Build duplicate_report_json
        duplicate_report_json = {
            "total_conversations_before": total_before,
            "total_duplicates_found": duplicates_removed,
            "total_conversations_after": total_after,
            "removed_duplicates": removed_duplicates_info,
        }

        # Build duplicate_report_md
        md_lines = [
            "# Exact Conversation Deduplication Report",
            "",
            "## Summary",
            f"- **Conversations before**: {total_before}",
            f"- **Duplicates removed**: {duplicates_removed}",
            f"- **Conversations after**: {total_after}",
            f"- **Duplicate percentage**: {duplicate_percentage:.2f}%",
            "",
            "## Details",
            "",
        ]

        if removed_duplicates_info:
            md_lines.extend([
                "| Kept ID | Removed ID | Kept Source | Removed Source |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for item in removed_duplicates_info:
                md_lines.append(
                    f"| {item['kept_conversation_id']} | "
                    f"{item['removed_conversation_id']} | "
                    f"{item['kept_source']} | "
                    f"{item['removed_source']} |"
                )
        else:
            md_lines.append("No duplicate conversations were found.")

        md_lines.append("")
        duplicate_report_md = "\n".join(md_lines)

        return deduplicated, manifest, duplicate_report_json, duplicate_report_md
