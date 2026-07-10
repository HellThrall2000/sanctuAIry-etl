"""Unit tests for the normalization framework and normalizers."""

from __future__ import annotations

import pytest

from cbt_companion.models.conversation import Conversation
from cbt_companion.models.raw_conversation import RawConversation, RawMessage
from cbt_companion.normalizers.empathetic_normalizer import EmpatheticNormalizer
from cbt_companion.normalizers.esconv_normalizer import ESConvNormalizer
from cbt_companion.normalizers.mental_health_normalizer import MentalHealthNormalizer
from cbt_companion.normalizers.registry import NormalizerRegistry


def test_empathetic_normalizer() -> None:
    # Arrange: Raw conversation representing EmpatheticDialogues
    raw_messages = [
        RawMessage(speaker_id=10, content="Hello there!"),
        RawMessage(speaker_id=10, content="How are you today?"),
        RawMessage(speaker_id=20, content="I'm feeling down."),
        RawMessage(speaker_id=10, content="Oh no, why?"),
    ]
    raw_conv = RawConversation(
        id="emp_1",
        source="facebook/empathetic_dialogues",
        messages=raw_messages,
        attributes={
            "context": "sadness",
            "prompt": "I lost my keys.",
            "split": "validation",
        },
    )

    normalizer = EmpatheticNormalizer()

    # Act
    conv = normalizer.normalize(raw_conv)

    # Assert
    assert isinstance(conv, Conversation)
    assert conv.id == "emp_1"
    assert conv.source == "facebook/empathetic_dialogues"
    assert conv.metadata.split == "validation"
    assert conv.metadata.source_dataset == "facebook/empathetic_dialogues"
    assert conv.metadata.original_id == "emp_1"
    assert conv.metadata.context == "sadness"

    # Message roles and content
    assert len(conv.messages) == 4
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "Hello there!"
    assert conv.messages[1].role == "user"
    assert conv.messages[1].content == "How are you today?"
    assert conv.messages[2].role == "assistant"
    assert conv.messages[2].content == "I'm feeling down."
    assert conv.messages[3].role == "user"
    assert conv.messages[3].content == "Oh no, why?"


def test_esconv_normalizer() -> None:
    # Arrange: Raw conversation representing ESConv
    raw_messages = [
        RawMessage(speaker_id="usr", content="Hi"),
        RawMessage(speaker_id="sys", content="Hello, how can I help you today?"),
        RawMessage(speaker_id="usr", content="I broke up with my partner."),
    ]
    raw_conv = RawConversation(
        id="es_123",
        source="thu-coai/esconv",
        messages=raw_messages,
        attributes={
            "emotion_type": "sadness",
            "problem_type": "breakup",
            "split": "train",
        },
    )

    normalizer = ESConvNormalizer()

    # Act
    conv = normalizer.normalize(raw_conv)

    # Assert
    assert isinstance(conv, Conversation)
    assert conv.id == "es_123"
    assert conv.source == "thu-coai/esconv"
    assert conv.metadata.split == "train"
    assert conv.metadata.source_dataset == "thu-coai/esconv"
    assert conv.metadata.original_id == "es_123"
    assert conv.metadata.context == "sadness"

    assert len(conv.messages) == 3
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "Hi"
    assert conv.messages[1].role == "assistant"
    assert conv.messages[1].content == "Hello, how can I help you today?"
    assert conv.messages[2].role == "user"
    assert conv.messages[2].content == "I broke up with my partner."


def test_mental_health_normalizer() -> None:
    # Arrange: Raw conversation representing Mental Health logs
    raw_messages = [
        RawMessage(speaker_id="Context", content="I'm feeling stressed."),
        RawMessage(speaker_id="Response", content="Try taking deep breaths."),
    ]
    raw_conv = RawConversation(
        id="mh_55",
        source="Amod/mental_health_counseling_conversations",
        messages=raw_messages,
        attributes={
            "split": "train",
        },
    )

    normalizer = MentalHealthNormalizer()

    # Act
    conv = normalizer.normalize(raw_conv)

    # Assert
    assert isinstance(conv, Conversation)
    assert conv.id == "mh_55"
    assert conv.source == "Amod/mental_health_counseling_conversations"
    assert conv.metadata.split == "train"
    assert conv.metadata.source_dataset == "Amod/mental_health_counseling_conversations"
    assert conv.metadata.original_id == "mh_55"
    assert conv.metadata.context is None

    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "I'm feeling stressed."
    assert conv.messages[1].role == "assistant"
    assert conv.messages[1].content == "Try taking deep breaths."


def test_normalizer_registry() -> None:
    # Act & Assert
    assert (
        NormalizerRegistry.get_normalizer_class("facebook/empathetic_dialogues")
        is EmpatheticNormalizer
    )
    assert NormalizerRegistry.get_normalizer_class("thu-coai/esconv") is ESConvNormalizer
    assert (
        NormalizerRegistry.get_normalizer_class("Amod/mental_health_counseling_conversations")
        is MentalHealthNormalizer
    )

    # Test registry fallback by case-insensitive name
    assert NormalizerRegistry.get_normalizer_class("esconv") is ESConvNormalizer
    assert NormalizerRegistry.get_normalizer_class("EMPATHETIC_DIALOGUES") is EmpatheticNormalizer

    with pytest.raises(KeyError):
        NormalizerRegistry.get_normalizer_class("nonexistent_dataset")
