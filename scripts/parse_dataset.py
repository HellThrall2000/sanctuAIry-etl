"""CLI script to parse a dataset and display ParseReport statistics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cbt_companion.core.registry import DatasetRegistry
from cbt_companion.parsers.registry import ParserRegistry
from datasets import DatasetDict, load_dataset

logger = logging.getLogger("cbt_companion.parse")

_DEFAULT_CONFIG = Path("configs/datasets.yaml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.parse_dataset",
        description="Parse a raw dataset split into raw conversations and print stats.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Registry key of the dataset to parse (e.g. 'empathetic_dialogues').",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="The dataset split to parse (default: 'train').",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to the datasets YAML config (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def load_raw_dataset(dataset_id: str) -> DatasetDict:
    """Load raw dataset from Hugging Face, mirroring inspector logic."""
    try:
        ds = load_dataset(dataset_id)
    except (RuntimeError, ValueError):
        logger.debug(
            "Standard load failed for '%s'; retrying with trust_remote_code=True.",
            dataset_id,
        )
        ds = load_dataset(dataset_id, trust_remote_code=True)

    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})
    return ds


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load registry
    config_path: Path = args.config
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    registry = DatasetRegistry.from_yaml(config_path)

    # Look up dataset
    dataset_key: str = args.dataset
    config = registry.get(dataset_key)
    if config is None:
        available = ", ".join(sorted(registry.keys()))
        logger.error(
            "Dataset '%s' not found in registry. Available keys: %s",
            dataset_key,
            available,
        )
        sys.exit(1)

    if not config.dataset_id:
        logger.error("Dataset '%s' configuration is missing 'dataset_id'.", dataset_key)
        sys.exit(1)

    # Get parser class from registry
    try:
        parser_class = ParserRegistry.get_parser_class(config.dataset_id)
    except KeyError as e:
        logger.error(str(e))
        sys.exit(1)

    # Load raw dataset
    logger.info("Loading dataset '%s' from Hugging Face...", config.dataset_id)
    try:
        ds_dict = load_raw_dataset(config.dataset_id)
    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", config.dataset_id, e)
        sys.exit(1)

    split_name: str = args.split
    if split_name not in ds_dict:
        available_splits = ", ".join(ds_dict.keys())
        logger.error(
            "Split '%s' not found in dataset. Available splits: %s",
            split_name,
            available_splits,
        )
        sys.exit(1)

    dataset_split = ds_dict[split_name]

    # Instantiate and parse
    parser_instance = parser_class()
    logger.info("Running parser '%s'...", parser_instance.__class__.__name__)
    try:
        report = parser_instance.parse(dataset_split, split_name)
    except Exception as e:
        logger.error("Parsing failed with exception: %s", e, exc_info=args.verbose)
        sys.exit(1)

    # Print statistics exactly in the requested format
    print("\n" + "=" * 40)
    print(f"Dataset:\n{report.parser_info.dataset_id}\n")
    print(f"Conversations parsed: {report.num_conversations}\n")
    print(f"Average turns: {report.avg_turns:.2f}\n")
    print(f"Maximum turns: {report.max_turns}\n")
    print(f"Minimum turns: {report.min_turns}\n")
    print(f"Skipped conversations: {report.skipped_conversations}")
    print("=" * 40)

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
