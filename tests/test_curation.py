"""Unit tests for the curation stage: quality rules, sampling and formatting."""

from __future__ import annotations

import json
from pathlib import Path

from cbt_companion.curation.chat_formatter import ChatFormatter
from cbt_companion.curation.curator import DatasetCurator
from cbt_companion.curation.quality_filter import QualityFilter
from cbt_companion.curation.sampler import SourceSampler
from cbt_companion.curation.text_cleaner import TextCleaner
from cbt_companion.curation.trimmer import ConversationTrimmer
from cbt_companion.curation.turn_normalizer import TurnNormalizer
from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message

LONG_REPLY = "That sounds genuinely difficult, and it makes sense that you feel this way."


def make_conversation(
    conv_id: str,
    *,
    source: str = "src",
    turns: list[tuple[str, str]] | None = None,
) -> Conversation:
    """Build a conversation from (role, content) pairs."""
    if turns is None:
        turns = [("user", "I feel low today"), ("assistant", LONG_REPLY)]
    return Conversation(
        id=conv_id,
        source=source,
        metadata=ConversationMetadata(),
        messages=[Message(role=r, content=c) for r, c in turns],
    )


# ── QualityFilter ──────────────────────────────────────────────────────────


def test_echo_turn_is_dropped() -> None:
    """Long enough to clear the length rule, so the echo rule is what fires."""
    conv = make_conversation(
        "echo",
        turns=[
            ("user", "I have not been able to sleep properly for three weeks."),
            ("assistant", "I have not been able to sleep properly for three weeks!"),
        ],
    )
    assert QualityFilter().evaluate(conv) == "assistant_echoes_user"


def test_length_rule_takes_precedence_over_echo() -> None:
    """A short echo is reported as too-short, since length is checked first."""
    conv = make_conversation(
        "short_echo",
        turns=[
            ("user", "This headache needs to go away."),
            ("assistant", "This headache needs to go away!"),
        ],
    )
    assert QualityFilter().evaluate(conv) == "final_response_too_short"


def test_substantive_reply_is_kept() -> None:
    assert QualityFilter().evaluate(make_conversation("ok")) is None


def test_short_final_response_is_dropped() -> None:
    conv = make_conversation(
        "short",
        turns=[("user", "I feel low today"), ("assistant", "Same here.")],
    )
    assert QualityFilter().evaluate(conv) == "final_response_too_short"


def test_occasional_backchannel_is_tolerated() -> None:
    """A short turn mid-dialogue must not discard a long counselling transcript."""
    conv = make_conversation(
        "backchannel",
        turns=[
            ("user", "I have been struggling at work lately"),
            ("assistant", "I see."),
            ("user", "My manager keeps criticising me in meetings"),
            ("assistant", LONG_REPLY),
            ("user", "It makes me want to quit"),
            ("assistant", LONG_REPLY),
        ],
    )
    assert QualityFilter().evaluate(conv) is None


def test_mostly_short_responses_are_dropped() -> None:
    conv = make_conversation(
        "terse",
        turns=[
            ("user", "I have been struggling at work lately"),
            ("assistant", "I see."),
            ("user", "My manager keeps criticising me"),
            ("assistant", "Oh no."),
            ("user", "It makes me want to quit"),
            ("assistant", LONG_REPLY),
        ],
    )
    assert QualityFilter().evaluate(conv) == "mostly_short_responses"


def test_conversation_emptied_by_cleaning_is_dropped() -> None:
    conv = make_conversation("empty", turns=[("user", "hi"), ("assistant", LONG_REPLY)])
    stripped = conv.model_copy(update={"messages": []})
    assert QualityFilter().evaluate(stripped) == "too_few_messages"


def test_filter_records_drop_reasons() -> None:
    conversations = [
        make_conversation("keep"),
        make_conversation(
            "drop", turns=[("user", "I am sad"), ("assistant", "Me too.")]
        ),
    ]
    quality = QualityFilter()
    kept = quality.filter(conversations)

    assert [c.id for c in kept] == ["keep"]
    assert quality.stats.num_seen == 2
    assert quality.stats.num_kept == 1
    assert quality.stats.drops_by_reason == {"final_response_too_short": 1}


# ── SourceSampler ──────────────────────────────────────────────────────────


def test_cap_keeps_the_most_substantive_conversations() -> None:
    conversations = [
        make_conversation(
            f"c{i}",
            source="chatty",
            turns=[("user", "hello"), ("assistant", "x" * (10 * i))],
        )
        for i in range(1, 6)
    ]
    kept = SourceSampler({"chatty": 2}, seed=1).sample(conversations)
    assert sorted(c.id for c in kept) == ["c4", "c5"]


def test_uncapped_and_missing_sources_are_untouched() -> None:
    conversations = [
        make_conversation("a", source="capped"),
        make_conversation("b", source="explicit_null"),
        make_conversation("c", source="not_in_caps"),
    ]
    kept = SourceSampler({"capped": 1, "explicit_null": None}, seed=1).sample(conversations)
    assert sorted(c.id for c in kept) == ["a", "b", "c"]


def test_sampling_is_deterministic_across_runs() -> None:
    conversations = [
        make_conversation(f"c{i}", source="s", turns=[("user", "hi"), ("assistant", "y" * i)])
        for i in range(1, 20)
    ]
    first = SourceSampler({"s": 5}, seed=7).sample(conversations)
    second = SourceSampler({"s": 5}, seed=7).sample(conversations)
    assert [c.id for c in first] == [c.id for c in second]


# ── ConversationTrimmer ────────────────────────────────────────────────────


def test_trailing_user_turns_are_trimmed_not_dropped() -> None:
    conv = make_conversation(
        "trailing",
        turns=[
            ("user", "I feel low today"),
            ("assistant", LONG_REPLY),
            ("user", "Thanks, that helps a lot"),
        ],
    )
    trimmed = ConversationTrimmer().trim(conv)
    assert [m.role for m in trimmed.messages] == ["user", "assistant"]
    assert QualityFilter().evaluate(trimmed) is None


def test_trailing_signoff_is_trimmed_back_to_substantive_turn() -> None:
    """A transcript closing on a pleasantry keeps its usable prefix."""
    conv = make_conversation(
        "signoff",
        turns=[
            ("user", "I have been struggling at work lately"),
            ("assistant", LONG_REPLY),
            ("user", "Thank you so much for listening to me"),
            ("assistant", "Good luck!"),
        ],
    )
    trimmed = ConversationTrimmer().trim(conv)
    assert [m.content for m in trimmed.messages] == [
        "I have been struggling at work lately",
        LONG_REPLY,
    ]
    assert QualityFilter().evaluate(trimmed) is None


def test_conversation_already_ending_on_assistant_is_unchanged() -> None:
    conv = make_conversation("ok")
    assert ConversationTrimmer().trim(conv) is conv


def test_conversation_without_assistant_turn_is_emptied() -> None:
    conv = make_conversation(
        "no_assistant",
        turns=[("user", "hello"), ("user", "anyone there")],
    )
    assert ConversationTrimmer().trim(conv).messages == []


def test_conversation_with_only_short_assistant_turns_is_emptied() -> None:
    conv = make_conversation(
        "all_short",
        turns=[("user", "I feel low today"), ("assistant", "Same here.")],
    )
    assert ConversationTrimmer().trim(conv).messages == []


# ── TurnNormalizer ─────────────────────────────────────────────────────────


def test_consecutive_same_role_turns_are_merged() -> None:
    conv = make_conversation(
        "runs",
        turns=[
            ("user", "I feel low"),
            ("user", "and I slept badly"),
            ("assistant", "That sounds hard."),
            ("assistant", "Tell me more."),
        ],
    )
    shaped = TurnNormalizer().normalize(conv)
    assert [m.role for m in shaped.messages] == ["user", "assistant"]
    sep = TurnNormalizer().separator
    assert shaped.messages[0].content == "I feel low" + sep + "and I slept badly"
    assert shaped.messages[1].content == "That sounds hard." + sep + "Tell me more."



def test_leading_assistant_turn_is_dropped() -> None:
    conv = make_conversation(
        "greeting",
        turns=[
            ("assistant", "Hello there. What would you like to talk about?"),
            ("user", "I just broke up with my partner"),
            ("assistant", LONG_REPLY),
        ],
    )
    shaped = TurnNormalizer().normalize(conv)
    assert [m.role for m in shaped.messages] == ["user", "assistant"]
    assert shaped.messages[0].content == "I just broke up with my partner"


def test_already_alternating_conversation_is_preserved() -> None:
    conv = make_conversation("clean")
    shaped = TurnNormalizer().normalize(conv)
    assert [m.role for m in shaped.messages] == ["user", "assistant"]


def test_normalizer_output_strictly_alternates_from_user() -> None:
    conv = make_conversation(
        "messy",
        turns=[
            ("assistant", "Hi there, how are you doing today?"),
            ("assistant", "Anything on your mind?"),
            ("user", "Work is hard"),
            ("user", "and I cannot sleep"),
            ("assistant", LONG_REPLY),
        ],
    )
    roles = [m.role for m in TurnNormalizer().normalize(conv).messages]
    assert roles == ["user", "assistant"]
    assert all(a != b for a, b in zip(roles, roles[1:]))


# ── ChatFormatter ──────────────────────────────────────────────────────────


def test_system_prompt_is_prepended() -> None:
    record = ChatFormatter("  Be kind.  ").format(make_conversation("c"))
    assert record["messages"][0] == {"role": "system", "content": "Be kind."}
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]


def test_no_system_turn_when_prompt_absent() -> None:
    record = ChatFormatter(None).format(make_conversation("c"))
    assert [m["role"] for m in record["messages"]] == ["user", "assistant"]


# ── DatasetCurator ─────────────────────────────────────────────────────────


def test_curator_writes_both_splits_and_reports(tmp_path: Path) -> None:
    conversations = [
        make_conversation(
            f"c{i}",
            source="facebook/empathetic_dialogues",
            turns=[("user", f"I feel low_comma_ day {i}"), ("assistant", LONG_REPLY)],
        )
        for i in range(20)
    ]
    # One conversation that must be filtered out by the echo rule.
    conversations.append(
        make_conversation(
            "echo",
            source="facebook/empathetic_dialogues",
            turns=[
                ("user", "I have not been able to sleep properly for three weeks."),
                ("assistant", "I have not been able to sleep properly for three weeks."),
            ],
        )
    )

    curator = DatasetCurator(
        cleaner=TextCleaner(),
        quality_filter=QualityFilter(),
        sampler=SourceSampler({}, seed=42),
        formatter=ChatFormatter("Be kind."),
        split_ratio=0.8,
        seed=42,
    )
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    report = curator.curate(conversations, train, validation)

    assert report.num_train + report.num_validation == 20
    assert report.num_train == 16
    assert report.quality.drops_by_reason == {"assistant_echoes_user": 1}
    assert report.counts_before == {"facebook/empathetic_dialogues": 21}

    lines = train.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 16
    record = json.loads(lines[0])
    assert record["messages"][0]["role"] == "system"
    # The cleaner ran before export, so no artifacts survive into training text.
    assert "_comma_" not in train.read_text(encoding="utf-8")
    assert "_comma_" not in validation.read_text(encoding="utf-8")


def test_curator_output_is_reproducible(tmp_path: Path) -> None:
    conversations = [
        make_conversation(f"c{i}", source="s") for i in range(30)
    ]

    def run(directory: Path) -> str:
        directory.mkdir()
        DatasetCurator(
            cleaner=TextCleaner(),
            quality_filter=QualityFilter(),
            sampler=SourceSampler({"s": 20}, seed=3),
            formatter=ChatFormatter(None),
            split_ratio=0.9,
            seed=3,
        ).curate(conversations, directory / "train.jsonl", directory / "val.jsonl")
        return (directory / "train.jsonl").read_text(encoding="utf-8")

    assert run(tmp_path / "a") == run(tmp_path / "b")
