"""Unit tests for the export_training_dataset script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_training_dataset import main


@pytest.fixture
def mock_dataset_file(tmp_path: Path) -> Path:
    """Creates a mock deduplicated dataset file with 20 distinct conversations."""
    input_file = tmp_path / "mock_deduplicated.jsonl"
    conversations = []
    for i in range(20):
        conversations.append({
            "id": f"conv_{i}",
            "source": f"source_{i % 3}",
            "metadata": {
                "split": "train",
                "source_dataset": "dummy",
                "original_id": f"orig_{i}",
                "context": "testing",
            },
            "messages": [
                {"role": "user", "content": f"Hello from user in conv {i} turn 1"},
                {"role": "assistant", "content": f"Hello from assistant in conv {i} turn 2"},
                {"role": "user", "content": f"Turn 3 in conv {i}"},
            ],
        })

    with input_file.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    return input_file


def test_export_split_ratio_and_fields(tmp_path: Path, mock_dataset_file: Path) -> None:
    """Verifies that the script splits 95/5 and only exports the messages field exactly as-is."""
    output_dir = tmp_path / "training_output"

    # Run the script with 20 conversations, 95% is 19 conversations
    main([
        "--input",
        str(mock_dataset_file),
        "--output-dir",
        str(output_dir),
        "--split-ratio",
        "0.95",
        "--seed",
        "42",
    ])

    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "validation.jsonl"

    assert train_file.exists()
    assert val_file.exists()

    # Read train.jsonl lines
    train_lines = train_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(train_lines) == 19

    # Read validation.jsonl lines
    val_lines = val_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(val_lines) == 1

    # Verify JSON structure and message preservation in train.jsonl
    for line in train_lines:
        data = json.loads(line)
        # ONLY messages field should be present
        assert list(data.keys()) == ["messages"]
        messages = data["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Check content is intact (no trimming or modifications)
        assert "Hello from user" in messages[0]["content"]
        assert "Hello from assistant" in messages[1]["content"]

    # Verify validation.jsonl
    val_data = json.loads(val_lines[0])
    assert list(val_data.keys()) == ["messages"]
    assert len(val_data["messages"]) == 3


def test_export_split_determinism(tmp_path: Path, mock_dataset_file: Path) -> None:
    """Verifies that the split and shuffling are deterministic when using seed 42."""
    out_dir_1 = tmp_path / "out_1"
    out_dir_2 = tmp_path / "out_2"
    out_dir_diff_seed = tmp_path / "out_diff"

    # Run first time with seed 42
    main([
        "--input",
        str(mock_dataset_file),
        "--output-dir",
        str(out_dir_1),
        "--seed",
        "42",
    ])

    # Run second time with seed 42
    main([
        "--input",
        str(mock_dataset_file),
        "--output-dir",
        str(out_dir_2),
        "--seed",
        "42",
    ])

    # Run third time with seed 100
    main([
        "--input",
        str(mock_dataset_file),
        "--output-dir",
        str(out_dir_diff_seed),
        "--seed",
        "100",
    ])

    # Verify exact contents are identical between out_1 and out_2
    train_1 = (out_dir_1 / "train.jsonl").read_text(encoding="utf-8")
    train_2 = (out_dir_2 / "train.jsonl").read_text(encoding="utf-8")
    val_1 = (out_dir_1 / "validation.jsonl").read_text(encoding="utf-8")
    val_2 = (out_dir_2 / "validation.jsonl").read_text(encoding="utf-8")

    assert train_1 == train_2
    assert val_1 == val_2

    # Verify different seed produces different shuffle/split contents
    train_diff = (out_dir_diff_seed / "train.jsonl").read_text(encoding="utf-8")
    assert train_1 != train_diff


def test_export_creates_directory_automatically(tmp_path: Path, mock_dataset_file: Path) -> None:
    """Verifies that nested output directories are created automatically by the script."""
    nested_dir = tmp_path / "level1" / "level2" / "level3"
    assert not nested_dir.exists()

    main([
        "--input",
        str(mock_dataset_file),
        "--output-dir",
        str(nested_dir),
    ])

    assert nested_dir.exists()
    assert (nested_dir / "train.jsonl").exists()
    assert (nested_dir / "validation.jsonl").exists()
