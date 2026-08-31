"""Unit tests for the training-set text cleaner."""

from __future__ import annotations

from cbt_companion.curation.text_cleaner import TextCleaner
from cbt_companion.models.conversation import Conversation, ConversationMetadata, Message


def test_comma_placeholder_is_replaced() -> None:
    cleaner = TextCleaner()
    result = cleaner.clean_text("Yeah_comma_ we grew up together_comma_ he is my friend")
    assert "_comma_" not in result
    assert result == "Yeah, we grew up together, he is my friend"
    assert cleaner.stats.replacements["_comma_"] == 2


def test_missing_space_after_sentence_punctuation() -> None:
    cleaner = TextCleaner()
    result = cleaner.clean_text("options that are available.It sounds like your partner is open")
    assert result == "options that are available. It sounds like your partner is open"


def test_missing_space_rule_does_not_split_numbers() -> None:
    """The rule needs a letter on both sides, so decimals stay intact."""
    cleaner = TextCleaner()
    assert cleaner.clean_text("I slept 6.5 hours") == "I slept 6.5 hours"
    assert cleaner.clean_text("it dropped 3.2 points") == "it dropped 3.2 points"


def test_missing_space_rule_leaves_correctly_spaced_text_alone() -> None:
    cleaner = TextCleaner()
    text = "That is hard. It sounds exhausting."
    assert cleaner.clean_text(text) == text
    assert "missing_space" not in cleaner.stats.replacements


def test_urls_are_stripped_when_enabled() -> None:
    cleaner = TextCleaner(strip_urls=True)
    result = cleaner.clean_text("Try this http://tinybuddha.com/blog/forgiving/ for tips")
    assert "http" not in result
    assert result == "Try this for tips"


def test_urls_are_preserved_when_disabled() -> None:
    cleaner = TextCleaner(strip_urls=False)
    result = cleaner.clean_text("Try this http://example.com/x for tips")
    assert "http://example.com/x" in result


def test_html_entities_are_unescaped() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean_text("you &amp; me") == "you & me"


def test_whitespace_is_collapsed_and_trimmed() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean_text("  too   many    spaces  ") == "too many spaces"
    assert cleaner.clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_conversation_rewrites_messages_and_counts_changes() -> None:
    conversation = Conversation(
        id="c1",
        source="facebook/empathetic_dialogues",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="I am sad_comma_ very sad"),
            Message(role="assistant", content="That sounds hard"),
        ],
    )
    cleaner = TextCleaner()
    cleaned = cleaner.clean_conversation(conversation)

    assert cleaned.messages[0].content == "I am sad, very sad"
    assert cleaned.messages[1].content == "That sounds hard"
    assert cleaner.stats.messages_seen == 2
    assert cleaner.stats.messages_changed == 1
    # The original conversation is untouched.
    assert conversation.messages[0].content == "I am sad_comma_ very sad"


def test_messages_emptied_by_cleaning_are_dropped() -> None:
    conversation = Conversation(
        id="c2",
        source="src",
        metadata=ConversationMetadata(),
        messages=[
            Message(role="user", content="http://example.com"),
            Message(role="assistant", content="Hello there"),
        ],
    )
    cleaned = TextCleaner(strip_urls=True).clean_conversation(conversation)
    assert len(cleaned.messages) == 1
    assert cleaned.messages[0].role == "assistant"
