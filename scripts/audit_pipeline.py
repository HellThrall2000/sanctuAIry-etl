"""Debugging script to run the pipeline and generate audit/breakdown reports."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from cbt_companion.analytics.pipeline_auditor import PipelineAuditor
from cbt_companion.core.registry import DatasetRegistry
from cbt_companion.filters.conversation_filter import ConversationFilter
from cbt_companion.mergers.conversation_merger import ConversationMerger
from cbt_companion.models.validation_report import ValidationConfig
from cbt_companion.normalizers.registry import NormalizerRegistry
from cbt_companion.parsers.registry import ParserRegistry
from cbt_companion.validation.conversation_validator import ConversationValidator
from datasets import DatasetDict, load_dataset

logger = logging.getLogger("cbt_companion.audit")


class PipelineConfigError(Exception):
    """Raised when pipeline configuration is missing, invalid or incomplete."""

    pass


def load_raw_dataset(dataset_id: str) -> DatasetDict:
    """Load raw dataset from Hugging Face."""
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


def parse_and_validate_config(config_path: Path) -> dict:
    """Read and validate the pipeline configuration file."""
    if not config_path.exists():
        raise PipelineConfigError(f"Required pipeline config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except Exception as e:
        raise PipelineConfigError(f"Failed to parse config YAML at {config_path}: {e}") from e

    if not isinstance(config, dict):
        raise PipelineConfigError(f"Invalid config at {config_path}: root must be a dictionary.")

    # Validate required top-level fields
    for field in ("datasets", "split", "validation", "sampling", "outputs"):
        if field not in config:
            raise PipelineConfigError(f"Missing required configuration field: '{field}'")

    return config


def run_audit(config: dict, dataset_registry: DatasetRegistry) -> None:
    """Execute pipeline and run the pipeline auditing stage."""
    split_name = config["split"]
    datasets_to_merge = []
    auditor = PipelineAuditor()

    # Track counts for table formatting
    loaded_counts = {}
    parsed_counts = {}
    normalized_counts = {}
    validated_counts = {}
    filtered_counts = {}

    # Stage 1-5: Process each dataset independently
    for dataset_id in config["datasets"]:
        # Resolve dataset configuration
        config_entry = None
        for entry in dataset_registry:
            if entry.dataset_id == dataset_id:
                config_entry = entry
                break
        if not config_entry:
            raise ValueError(f"Dataset ID '{dataset_id}' could not be resolved.")

        print(f"[*] Auditing dataset: {dataset_id}")

        # 1. Load dataset
        logger.info("[%s] Loading split '%s'...", dataset_id, split_name)
        ds_dict = load_raw_dataset(dataset_id)
        if split_name not in ds_dict:
            raise ValueError(f"Dataset '{dataset_id}' is missing requested split '{split_name}'")
        dataset_split = ds_dict[split_name]
        loaded_counts[dataset_id] = len(dataset_split)

        # 2. Parse raw rows into RawConversations
        parser_class = ParserRegistry.get_parser_class(dataset_id)
        parser = parser_class()
        logger.info("[%s] Parsing raw rows...", dataset_id)
        parse_report = parser.parse(dataset_split, split_name)
        parsed_counts[dataset_id] = parse_report.num_conversations

        # 3. Normalize to canonical Conversations
        normalizer_class = NormalizerRegistry.get_normalizer_class(dataset_id)
        normalizer = normalizer_class()
        logger.info("[%s] Normalizing turns...", dataset_id)
        normalized_conversations = [normalizer.normalize(rc) for rc in parse_report.conversations]
        normalized_counts[dataset_id] = len(normalized_conversations)

        # 4. Validate canonical structures in-memory
        val_config = ValidationConfig()
        validator = ConversationValidator(config=val_config)
        logger.info("[%s] Running structural validation...", dataset_id)
        val_report = validator.validate_dataset(normalized_conversations)
        validated_counts[dataset_id] = len(normalized_conversations)

        # Record validation failures details
        auditor.record_validation_failures(normalized_conversations, val_report)

        # 5. Filter out invalid conversations
        filter_inst = ConversationFilter()
        logger.info("[%s] Filtering invalid conversations...", dataset_id)
        filter_report = filter_inst.filter(normalized_conversations, val_report)
        filtered_counts[dataset_id] = filter_report.num_kept

        datasets_to_merge.append(filter_report.filtered_conversations)
        print(f"[+] Dataset '{dataset_id}' audited successfully.")

    # Record Stages 1-5 counts in Auditor
    auditor.record_stage("Loaded", loaded_counts)
    auditor.record_stage("Parsed", parsed_counts)
    auditor.record_stage("Normalized", normalized_counts)
    auditor.record_stage("Validated", validated_counts)
    auditor.record_stage("Filtered", filtered_counts)

    # 6. Merge valid datasets
    print("[*] Merging all filtered datasets...")
    merger = ConversationMerger()
    merge_report = merger.merge(datasets_to_merge)
    print(f"[+] Merged dataset has {merge_report.total_conversations} conversations.")

    # Record Merged counts in Auditor
    # Map from config dataset ID -> count
    merged_counts = {}
    for dataset_id in config["datasets"]:
        merged_counts[dataset_id] = merge_report.contribution_counts.get(dataset_id, 0)
    auditor.record_stage("Merged", merged_counts)

    # Export audit reports
    print("[*] Exporting audit and breakdown reports...")
    audit_md_path = "outputs/reports/pipeline_audit.md"
    breakdown_md_path = "outputs/reports/validation_breakdown.md"
    examples_dir = "outputs/review/validation_examples"

    auditor.export_audit(
        audit_path=audit_md_path,
        breakdown_path=breakdown_md_path,
        examples_dir=examples_dir,
    )

    print("\n" + "=" * 40)
    print("PIPELINE AUDIT COMPLETED SUCCESSFULLY!")
    print(f"Audit Summary:        {audit_md_path}")
    print(f"Validation Breakdown: {breakdown_md_path}")
    print(f"Validation Examples:  {examples_dir}/")
    print("=" * 40 + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="scripts.audit_pipeline",
        description="Run pipeline and generate audit/debug reports.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline_config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args(argv)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Parse and validate configurations
    try:
        config = parse_and_validate_config(args.config)
        dataset_registry = DatasetRegistry.from_yaml(Path("configs/datasets.yaml"))
    except (PipelineConfigError, FileNotFoundError) as e:
        logger.error("Configuration Error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected configuration initialization failure: %s", e)
        sys.exit(1)

    # 2. Execute pipeline audit
    try:
        run_audit(config, dataset_registry)
    except Exception as e:
        logger.error("Pipeline audit failed: %s", e, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
