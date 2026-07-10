"""CLI entry point for dataset inspection.

Usage::

    python -m cbt_companion.inspect --dataset empathetic_dialogues
    python -m cbt_companion.inspect --dataset esconv
    python -m cbt_companion.inspect --dataset mental_health_counseling_conversations
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cbt_companion.core.registry import DatasetRegistry
from cbt_companion.inspect.inspector import DatasetInspector
from cbt_companion.inspect.report import MarkdownReportGenerator

logger = logging.getLogger("cbt_companion.inspect")

# ── Defaults ─────────────────────────────────────────────────

_DEFAULT_CONFIG = Path("configs/datasets.yaml")
_DEFAULT_OUTPUT = Path("outputs/inspection")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbt_companion.inspect",
        description="Inspect a raw dataset and generate a Markdown report.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Registry key of the dataset to inspect (e.g. 'esconv').",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to the datasets YAML config (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Directory where the report will be saved (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of random samples to include per split (default: 5).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Logging
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
    logger.info("Loaded registry with %d dataset(s).", len(registry))

    # Look up requested dataset
    dataset_key: str = args.dataset
    config = registry.get(dataset_key)
    if config is None:
        available = ", ".join(sorted(registry.keys()))
        logger.error(
            "Dataset '%s' not found in registry. Available: %s",
            dataset_key,
            available,
        )
        sys.exit(1)

    # Inspect
    inspector = DatasetInspector(config, sample_count=args.samples)
    result = inspector.inspect()

    # Report
    generator = MarkdownReportGenerator(result)
    report_path = generator.save(args.output)
    print(f"\n[SUCCESS] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
