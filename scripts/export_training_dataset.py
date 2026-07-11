"""CLI script to prepare and export the final dataset for supervised fine-tuning."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger("cbt_companion.export_training")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.export_training_dataset",
        description="Prepare final fine-tuning dataset by shuffling, splitting and exporting.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/final/deduplicated_dataset.jsonl"),
        help="Input deduplicated JSONL path (default: outputs/final/deduplicated_dataset.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training"),
        help="Directory to write the split outputs (default: outputs/training).",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.95,
        help="Split ratio for training set (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffle.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path: Path = args.input
    output_dir: Path = args.output_dir
    split_ratio: float = args.split_ratio
    seed: int = args.seed

    if not input_path.exists():
        logger.error("Input dataset file not found: %s", input_path)
        sys.exit(1)

    logger.info("Reading input dataset from %s...", input_path)
    conversations = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON line: %s", e)
                sys.exit(1)

    total_convs = len(conversations)
    logger.info("Loaded %d conversations.", total_convs)

    if total_convs == 0:
        logger.error("No conversations found in the input dataset.")
        sys.exit(1)

    # Deterministic shuffle
    logger.info("Shuffling conversations using seed %d...", seed)
    rng = random.Random(seed)
    rng.shuffle(conversations)

    # 95/5 train/val split
    split_idx = int(total_convs * split_ratio)
    train_convs = conversations[:split_idx]
    val_convs = conversations[split_idx:]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    logger.info("Writing training split to %s...", train_path)
    with train_path.open("w", encoding="utf-8") as f:
        for conv in train_convs:
            transformed = {"messages": conv["messages"]}
            f.write(json.dumps(transformed, ensure_ascii=False) + "\n")

    logger.info("Writing validation split to %s...", val_path)
    with val_path.open("w", encoding="utf-8") as f:
        for conv in val_convs:
            transformed = {"messages": conv["messages"]}
            f.write(json.dumps(transformed, ensure_ascii=False) + "\n")

    # Console summary
    print(f"Training conversations: {len(train_convs)}")
    print(f"Validation conversations: {len(val_convs)}")
    print(f"Output directory: {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
