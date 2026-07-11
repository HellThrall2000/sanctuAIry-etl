"""Unit tests for the analyze_tokens script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.analyze_tokens import (
    calculate_percentile,
    get_bin_name,
    get_recommended_context_length,
    main,
)


def test_calculate_percentile() -> None:
    # Arrange
    data = [10, 20, 30, 40, 50]

    # Act & Assert
    assert calculate_percentile(data, 0) == 10
    assert calculate_percentile(data, 50) == 30
    assert calculate_percentile(data, 100) == 50

    # Interpolation checks
    # len is 5, index is 4 * 0.25 = 1.0 (20)
    assert calculate_percentile(data, 25.0) == 20
    # index is 4 * 0.75 = 3.0 (40)
    assert calculate_percentile(data, 75.0) == 40
    # index is 4 * 0.9 = 3.6 (between 40 and 50) -> 40 + 0.6 * 10 = 46
    assert calculate_percentile(data, 90.0) == 46

    # Empty list edge case
    assert calculate_percentile([], 95.0) == 0


def test_get_recommended_context_length() -> None:
    # Arrange & Act & Assert
    assert get_recommended_context_length(500) == 1024
    assert get_recommended_context_length(1024) == 1024
    assert get_recommended_context_length(1025) == 2048
    assert get_recommended_context_length(2048) == 2048
    assert get_recommended_context_length(2049) == 4096
    assert get_recommended_context_length(4096) == 4096
    assert get_recommended_context_length(4097) == 8192
    assert get_recommended_context_length(8192) == 8192
    assert get_recommended_context_length(8193) == 16384
    assert get_recommended_context_length(20000) == 32768


def test_get_bin_name() -> None:
    # Arrange & Act & Assert
    assert get_bin_name(50) == "0-128"
    assert get_bin_name(128) == "0-128"
    assert get_bin_name(129) == "129-256"
    assert get_bin_name(256) == "129-256"
    assert get_bin_name(257) == "257-512"
    assert get_bin_name(512) == "257-512"
    assert get_bin_name(513) == "513-1024"
    assert get_bin_name(1024) == "513-1024"
    assert get_bin_name(1025) == "1025-2048"
    assert get_bin_name(2048) == "1025-2048"
    assert get_bin_name(2049) == "2049-4096"
    assert get_bin_name(4096) == "2049-4096"
    assert get_bin_name(4097) == "4097-8192"
    assert get_bin_name(8192) == "4097-8192"
    assert get_bin_name(8193) == "8193+"


class DummyTokenizer:
    """Mock tokenizer simulating AutoTokenizer logic."""

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = False
    ) -> str:
        # Simple serialization
        return " | ".join(f"{msg['role']}: {msg['content']}" for msg in messages)

    def encode(self, text: str) -> list[int]:
        # Emits a simulated token for every character (excluding spaces)
        chars = text.replace(" ", "")
        return [1] * len(chars)


@pytest.fixture
def mock_training_file(tmp_path: Path) -> Path:
    """Creates a mock training JSONL file with known configurations."""
    input_file = tmp_path / "mock_train.jsonl"
    conversations = [
        {
            # 2 messages, total characters in dummy template: 31
            # 'user: short | assistant: hello'
            "messages": [
                {"role": "user", "content": "short"},
                {"role": "assistant", "content": "hello"},
            ]
        },
        {
            # 4 messages, total characters: 69
            # 'user: first | assistant: second | user: third | assistant: fourth'
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
                {"role": "assistant", "content": "fourth"},
            ]
        },
        {
            # 2 messages, long content, total characters: 1500
            "messages": [
                {"role": "user", "content": "A" * 1400},
                {"role": "assistant", "content": "B" * 80},
            ]
        },
    ]

    with input_file.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    return input_file


@patch("transformers.AutoTokenizer.from_pretrained")
def test_analyze_tokens_script_success(
    mock_from_pretrained: MagicMock, tmp_path: Path, mock_training_file: Path
) -> None:
    """Verifies end-to-end analyze_tokens.py behavior with mock tokenizer."""
    # Arrange
    dummy_tokenizer = DummyTokenizer()
    mock_from_pretrained.return_value = dummy_tokenizer

    output_dir = tmp_path / "reports"

    # Act
    main([
        "--input",
        str(mock_training_file),
        "--output-dir",
        str(output_dir),
        "--model-id",
        "mock-gemma",
    ])

    # Assert
    # 1. Output files exist
    json_path = output_dir / "token_statistics.json"
    md_path = output_dir / "token_statistics.md"
    assert json_path.exists()
    assert md_path.exists()

    # 2. Verify JSON report contents
    with json_path.open("r", encoding="utf-8") as f:
        report_data = json.load(f)

    assert report_data["total_conversations"] == 3
    # 26 + 55 + 1496 = 1577 tokens
    assert report_data["total_tokens"] == 1577
    assert report_data["average_tokens"] == pytest.approx(525.67, 0.01)
    assert report_data["median_tokens"] == 55.0
    assert report_data["minimum_tokens"] == 26
    assert report_data["maximum_tokens"] == 1496
    # average turns: (2 + 4 + 2) / 3 = 2.67
    assert report_data["average_turns_per_conversation"] == pytest.approx(2.67, 0.01)
    # p99 check: sorted [26, 55, 1496] -> linear index 2 * 0.99 = 1.98
    # 55 + 0.98 * (1496 - 55) = 1467.18 -> 1467
    assert report_data["percentile_99_tokens"] == 1467
    # 1467 <= 2048 -> recommended context should be 2048
    assert report_data["recommended_context_length"] == 2048

    # check threshold stats
    assert report_data["conversations_over_1024"] == 1
    assert report_data["conversations_over_2048"] == 0

    # check histogram classifications
    hist = report_data["histogram"]
    assert hist["0-128"] == 2  # 26, 55
    assert hist["1025-2048"] == 1  # 1496

    # 3. Verify markdown report formatting
    md_content = md_path.read_text(encoding="utf-8")
    assert "# Gemma Token Analysis Report" in md_content
    assert "**Model ID**: `mock-gemma`" in md_content
    assert "- **Total tokens**: 1577" in md_content
    assert "- **Recommended context length**: 2048" in md_content
    assert "- **Average turns per conversation**: 2.67" in md_content
    assert "| 0-128 | 2 | 66.67% |" in md_content
    assert "| 1025-2048 | 1 | 33.33% |" in md_content
