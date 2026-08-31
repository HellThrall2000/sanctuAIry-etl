"""Main script to execute the end-to-end dataset processing pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from cbt_companion.analytics.canonical_exporter import CanonicalDatasetExporter
from cbt_companion.analytics.deduplicator import ConversationDeduplicator
from cbt_companion.analytics.review_exporter import ReviewExporter
from cbt_companion.analytics.statistics_tool import StatisticsTool
from cbt_companion.core.registry import DatasetRegistry
from cbt_companion.curation.chat_formatter import ChatFormatter
from cbt_companion.curation.curator import DatasetCurator
from cbt_companion.curation.quality_filter import QualityFilter
from cbt_companion.curation.report_writer import CurationReportWriter
from cbt_companion.curation.sampler import SourceSampler
from cbt_companion.curation.text_cleaner import TextCleaner
from cbt_companion.curation.trimmer import ConversationTrimmer
from cbt_companion.filters.conversation_filter import ConversationFilter
from cbt_companion.filters.registry import FilterRegistry
from cbt_companion.mergers.conversation_merger import ConversationMerger
from cbt_companion.models.validation_report import ValidationConfig
from cbt_companion.normalizers.registry import NormalizerRegistry
from cbt_companion.parsers.registry import ParserRegistry
from cbt_companion.validation.conversation_validator import ConversationValidator
from cbt_companion.validation.registry import ValidatorRegistry
from datasets import DatasetDict, load_dataset

logger = logging.getLogger("cbt_companion.pipeline")


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
    """Read and validate the pipeline configuration file.

    Args:
        config_path: Path to the pipeline config YAML file.

    Returns:
        The validated configuration dictionary.

    Raises:
        PipelineConfigError: If config file is missing or contains invalid settings.
    """
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
    for field in ("datasets", "split", "validation", "sampling", "outputs", "dataset_version"):
        if field not in config:
            raise PipelineConfigError(f"Missing required configuration field: '{field}'")

    datasets = config["datasets"]
    if not isinstance(datasets, list) or not all(isinstance(d, str) for d in datasets):
        raise PipelineConfigError("'datasets' must be a list of strings (Hugging Face IDs).")
    if not datasets:
        raise PipelineConfigError("'datasets' list cannot be empty.")

    if not isinstance(config["split"], str) or not config["split"].strip():
        raise PipelineConfigError("'split' must be a non-empty string.")

    validation = config["validation"]
    if not isinstance(validation, dict):
        raise PipelineConfigError("'validation' must be a dictionary.")

    sampling = config["sampling"]
    if not isinstance(sampling, dict):
        raise PipelineConfigError("'sampling' must be a dictionary.")
    for f in ("sample_size", "seed"):
        if f not in sampling or not isinstance(sampling[f], int):
            raise PipelineConfigError(f"'sampling.{f}' must be an integer.")

    outputs = config["outputs"]
    if not isinstance(outputs, dict):
        raise PipelineConfigError("'outputs' must be a dictionary.")
    for f in (
        "statistics_json",
        "statistics_md",
        "review_sample_json",
        "review_sample_md",
        "canonical_dataset_jsonl",
        "deduplicated_dataset_jsonl",
        "manifest_json",
        "duplicate_report_json",
        "duplicate_report_md",
        "train_jsonl",
        "validation_jsonl",
        "curation_report_json",
        "curation_report_md",
    ):
        if f not in outputs or not isinstance(outputs[f], str) or not outputs[f].strip():
            raise PipelineConfigError(f"'outputs.{f}' must be a non-empty string filepath.")

    if not isinstance(config["dataset_version"], str) or not config["dataset_version"].strip():
        raise PipelineConfigError("'dataset_version' must be a non-empty string.")

    curation = config.get("curation")
    if curation is not None:
        if not isinstance(curation, dict):
            raise PipelineConfigError("'curation' must be a dictionary.")

        ratio = curation.get("split_ratio", 0.95)
        if not isinstance(ratio, (int, float)) or not 0.0 < float(ratio) < 1.0:
            raise PipelineConfigError("'curation.split_ratio' must be a float between 0 and 1.")

        if not isinstance(curation.get("seed", 42), int):
            raise PipelineConfigError("'curation.seed' must be an integer.")

        caps = curation.get("source_caps", {})
        if not isinstance(caps, dict):
            raise PipelineConfigError("'curation.source_caps' must be a dictionary.")
        for source, cap in caps.items():
            if cap is not None and (not isinstance(cap, int) or cap < 1):
                raise PipelineConfigError(
                    f"'curation.source_caps.{source}' must be a positive integer or null."
                )

        prompt_file = curation.get("system_prompt_file")
        if prompt_file is not None and not isinstance(prompt_file, str):
            raise PipelineConfigError("'curation.system_prompt_file' must be a string path.")

    return config


def verify_dry_run(config: dict, dataset_registry: DatasetRegistry) -> None:
    """Validate all components and print the execution plan without running pipeline."""
    logger.info("Starting dry-run validation of registries and configurations...")

    # 1. Verify datasets can be resolved in registry
    resolved_entries = []
    for dataset_id in config["datasets"]:
        config_entry = None
        for entry in dataset_registry:
            if entry.dataset_id == dataset_id:
                config_entry = entry
                break
        if not config_entry:
            raise PipelineConfigError(
                f"Dataset ID '{dataset_id}' specified in pipeline config "
                f"could not be resolved in datasets registry (configs/datasets.yaml)."
            )
        resolved_entries.append(config_entry)

    # 2. Verify dataset-specific parser and normalizer registrations
    for entry in resolved_entries:
        try:
            parser_class = ParserRegistry.get_parser_class(entry.dataset_id)
        except KeyError as e:
            raise PipelineConfigError(
                f"Dry-run failure: No parser registered for dataset ID "
                f"'{entry.dataset_id}'. Details: {e}"
            )

        try:
            normalizer_class = NormalizerRegistry.get_normalizer_class(entry.dataset_id)
        except KeyError as e:
            raise PipelineConfigError(
                f"Dry-run failure: No normalizer registered for dataset ID "
                f"'{entry.dataset_id}'. Details: {e}"
            )

        logger.debug(
            "Resolved components for %s: Parser=%s, Normalizer=%s",
            entry.dataset_id,
            parser_class.__name__,
            normalizer_class.__name__,
        )

    # 3. Verify general validation, filter, and merger components
    try:
        validator_class = ValidatorRegistry.get_validator_class("conversation")
        if validator_class is not ConversationValidator:
            raise PipelineConfigError(f"Unexpected validator class resolved: {validator_class}")
    except KeyError as e:
        raise PipelineConfigError(f"Dry-run failure: Validator class resolution failed: {e}")

    try:
        filter_class = FilterRegistry.get_filter_class("conversation")
        if filter_class is not ConversationFilter:
            raise PipelineConfigError(f"Unexpected filter class resolved: {filter_class}")
    except KeyError as e:
        raise PipelineConfigError(f"Dry-run failure: Filter class resolution failed: {e}")

    # Output execution plan
    plan_lines = [
        "",
        "=" * 60,
        "PIPELINE DRY-RUN EXECUTION PLAN (VALID)",
        "-" * 60,
        f"Target Split: {config['split']}",
        "Datasets to process:",
    ]
    for entry in resolved_entries:
        p_name = ParserRegistry.get_parser_class(entry.dataset_id).__name__
        n_name = NormalizerRegistry.get_normalizer_class(entry.dataset_id).__name__
        plan_lines.append(f"  - {entry.dataset_id} ({p_name} -> {n_name})")

    plan_lines.extend([
        "",
        "Validation settings:",
        "  - Purely structural validation enabled",
        "",
        "Sampling parameters:",
        f"  - sample_size: {config['sampling']['sample_size']} conversations",
        f"  - seed: {config['sampling']['seed']}",
        "",
        "Configured outputs:",
        f"  - Canonical JSONL:     {config['outputs']['canonical_dataset_jsonl']}",
        f"  - Deduplicated JSONL:  {config['outputs']['deduplicated_dataset_jsonl']}",
        f"  - Manifest JSON:       {config['outputs']['manifest_json']}",
        f"  - Duplicate JSON Rep:  {config['outputs']['duplicate_report_json']}",
        f"  - Duplicate MD Rep:    {config['outputs']['duplicate_report_md']}",
        f"  - Statistics JSON:     {config['outputs']['statistics_json']}",
        f"  - Statistics MD:       {config['outputs']['statistics_md']}",
        f"  - Review JSON:         {config['outputs']['review_sample_json']}",
        f"  - Review MD:           {config['outputs']['review_sample_md']}",
        "",
        "Curation settings:",
    ])

    curation = config.get("curation") or {}
    if not curation.get("enabled", True):
        plan_lines.append("  - DISABLED (no training splits will be written)")
    else:
        caps = curation.get("source_caps") or {}
        quality = curation.get("quality") or {}
        plan_lines.extend([
            f"  - system_prompt_file: {curation.get('system_prompt_file') or '(none)'}",
            f"  - split_ratio: {curation.get('split_ratio', 0.95)}",
            f"  - min_assistant_chars: {quality.get('min_assistant_chars', 40)}",
            f"  - echo_similarity_threshold: {quality.get('echo_similarity_threshold', 0.8)}",
            "  - source caps:",
        ])
        for source in config["datasets"]:
            cap = caps.get(source)
            plan_lines.append(f"      {source}: {cap if cap is not None else 'uncapped'}")
        plan_lines.extend([
            f"  - Train JSONL:         {config['outputs']['train_jsonl']}",
            f"  - Validation JSONL:    {config['outputs']['validation_jsonl']}",
            f"  - Curation Rep JSON:   {config['outputs']['curation_report_json']}",
            f"  - Curation Rep MD:     {config['outputs']['curation_report_md']}",
        ])

    plan_lines.extend([
        "=" * 60,
        "",
    ])

    print("\n".join(plan_lines))


def build_curator(config: dict) -> DatasetCurator | None:
    """Assemble the curation stage from configuration.

    Args:
        config: The validated pipeline configuration.

    Returns:
        A configured DatasetCurator, or None when curation is disabled.

    Raises:
        PipelineConfigError: If the configured system prompt file is missing.
    """
    curation = config.get("curation") or {}
    if not curation.get("enabled", True):
        return None

    system_prompt = None
    prompt_file = curation.get("system_prompt_file")
    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise PipelineConfigError(f"Curation system prompt file not found: {prompt_path}")
        system_prompt = prompt_path.read_text(encoding="utf-8")

    cleaning = curation.get("cleaning") or {}
    quality = curation.get("quality") or {}
    seed = curation.get("seed", 42)

    return DatasetCurator(
        cleaner=TextCleaner(strip_urls=cleaning.get("strip_urls", True)),
        quality_filter=QualityFilter(
            min_assistant_chars=quality.get("min_assistant_chars", 40),
            max_short_assistant_ratio=quality.get("max_short_assistant_ratio", 0.5),
            echo_similarity_threshold=quality.get("echo_similarity_threshold", 0.8),
            min_messages=quality.get("min_messages", 2),
        ),
        trimmer=ConversationTrimmer(
            min_final_assistant_chars=quality.get("min_assistant_chars", 40),
        ),
        sampler=SourceSampler(curation.get("source_caps") or {}, seed=seed),
        formatter=ChatFormatter(system_prompt),
        split_ratio=float(curation.get("split_ratio", 0.95)),
        seed=seed,
    )


def execute_pipeline(config: dict, dataset_registry: DatasetRegistry) -> None:
    """Execute the pipeline stages in order, stopping on unrecoverable errors."""
    split_name = config["split"]
    datasets_to_merge = []

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

        print(f"[*] Processing dataset: {dataset_id}")

        # 1. Load dataset
        logger.info("[%s] Loading split '%s'...", dataset_id, split_name)
        ds_dict = load_raw_dataset(dataset_id)
        if split_name not in ds_dict:
            raise ValueError(f"Dataset '{dataset_id}' is missing requested split '{split_name}'")
        dataset_split = ds_dict[split_name]
        logger.info("[%s] Dataset split loaded successfully.", dataset_id)

        # 2. Parse raw rows into RawConversations
        parser_class = ParserRegistry.get_parser_class(dataset_id)
        parser = parser_class()
        logger.info("[%s] Parsing raw rows using %s...", dataset_id, parser.__class__.__name__)
        parse_report = parser.parse(dataset_split, split_name)
        logger.info(
            "[%s] Parsed %d conversations (skipped %d).",
            dataset_id,
            parse_report.num_conversations,
            parse_report.skipped_conversations,
        )

        # 3. Normalize to canonical Conversations
        normalizer_class = NormalizerRegistry.get_normalizer_class(dataset_id)
        normalizer = normalizer_class()
        logger.info("[%s] Normalizing turns using %s...", dataset_id, normalizer.__class__.__name__)
        normalized_conversations = [normalizer.normalize(rc) for rc in parse_report.conversations]

        # 4. Validate canonical structures in-memory
        val_config = ValidationConfig()
        validator = ConversationValidator(config=val_config)
        logger.info("[%s] Running structural validation...", dataset_id)
        val_report = validator.validate_dataset(normalized_conversations)
        logger.info(
            "[%s] Validation complete. Valid: %d, Invalid: %d",
            dataset_id,
            val_report.num_valid,
            val_report.num_invalid,
        )

        # 5. Filter out invalid conversations (without removing from disk)
        filter_inst = ConversationFilter()
        logger.info("[%s] Filtering invalid conversations...", dataset_id)
        filter_report = filter_inst.filter(normalized_conversations, val_report)
        logger.info("[%s] Retained %d valid conversations.", dataset_id, filter_report.num_kept)

        datasets_to_merge.append(filter_report.filtered_conversations)
        print(f"[+] Dataset '{dataset_id}' processing completed successfully.")

    # 6. Merge valid datasets
    print("[*] Merging all filtered datasets...")
    merger = ConversationMerger()
    # Pass datasets sequence to the merger
    merge_report = merger.merge(datasets_to_merge)
    print(f"[+] Merged dataset has {merge_report.total_conversations} conversations.")

    # 7. Export canonical dataset to JSONL
    canonical_path = config["outputs"]["canonical_dataset_jsonl"]
    print("[*] Exporting canonical dataset...")
    num_exported = CanonicalDatasetExporter.export(
        merge_report.merged_conversations,
        canonical_path,
    )
    file_size_bytes = Path(canonical_path).stat().st_size
    if file_size_bytes >= 1_048_576:
        size_str = f"{file_size_bytes / 1_048_576:.2f} MB"
    else:
        size_str = f"{file_size_bytes / 1024:.2f} KB"
    print(f"[+] Exported {num_exported} conversations to {canonical_path} ({size_str})")

    # 8. Deduplicate canonical dataset
    print("[*] Performing exact conversation deduplication...")
    dedup_results = ConversationDeduplicator.deduplicate(
        merge_report.merged_conversations,
        dataset_version=config["dataset_version"],
    )
    deduplicated_convs, manifest, dup_report_json, dup_report_md = dedup_results

    # Write deduplicated JSONL
    dedup_path = config["outputs"]["deduplicated_dataset_jsonl"]
    print("[*] Exporting deduplicated dataset...")
    num_dedup_exported = CanonicalDatasetExporter.export(deduplicated_convs, dedup_path)
    dedup_size_bytes = Path(dedup_path).stat().st_size
    if dedup_size_bytes >= 1_048_576:
        dedup_size_str = f"{dedup_size_bytes / 1_048_576:.2f} MB"
    else:
        dedup_size_str = f"{dedup_size_bytes / 1024:.2f} KB"
    print(f"[+] Exported {num_dedup_exported} conversations to {dedup_path} ({dedup_size_str})")

    # Write manifest.json
    manifest_path = config["outputs"]["manifest_json"]
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Write duplicate_report.json
    rep_json_path = config["outputs"]["duplicate_report_json"]
    os.makedirs(os.path.dirname(os.path.abspath(rep_json_path)), exist_ok=True)
    with open(rep_json_path, "w", encoding="utf-8") as f:
        json.dump(dup_report_json, f, indent=2, ensure_ascii=False)

    # Write duplicate_report.md
    rep_md_path = config["outputs"]["duplicate_report_md"]
    os.makedirs(os.path.dirname(os.path.abspath(rep_md_path)), exist_ok=True)
    with open(rep_md_path, "w", encoding="utf-8") as f:
        f.write(dup_report_md)

    # Update MergeReport for downstream analytics to reflect deduplicated dataset
    merge_report.merged_conversations = deduplicated_convs
    merge_report.total_conversations = len(deduplicated_convs)
    new_counts = {}
    for conv in deduplicated_convs:
        new_counts[conv.source] = new_counts.get(conv.source, 0) + 1
    merge_report.contribution_counts = new_counts

    # 9. Compute statistics
    print("[*] Computing dataset statistics...")
    stats_report = StatisticsTool.compute_statistics(merge_report)

    # 10. Deterministic sampling & exports
    print("[*] Generating review sample and exporting reports...")
    sample_size = config["sampling"]["sample_size"]
    seed = config["sampling"]["seed"]
    review_sample = StatisticsTool.generate_sample(merge_report, sample_size=sample_size, seed=seed)

    # Export statistics JSON and Markdown reports
    ReviewExporter.export_statistics(
        stats_report,
        config["outputs"]["statistics_json"],
        config["outputs"]["statistics_md"],
    )

    # Export review sample JSON and Markdown transcriptions
    ReviewExporter.export_review_sample(
        review_sample,
        config["outputs"]["review_sample_json"],
        config["outputs"]["review_sample_md"],
    )

    # 11. Curate the deduplicated corpus into supervised fine-tuning splits
    curator = build_curator(config)
    curation_report = None
    if curator is None:
        print("[*] Curation stage disabled; skipping training export.")
    else:
        print("[*] Curating training splits (clean -> quality -> rebalance -> split)...")
        curation_report = curator.curate(
            deduplicated_convs,
            Path(config["outputs"]["train_jsonl"]),
            Path(config["outputs"]["validation_jsonl"]),
        )
        CurationReportWriter.export(
            curation_report,
            Path(config["outputs"]["curation_report_json"]),
            Path(config["outputs"]["curation_report_md"]),
        )
        print(
            f"[+] Curated {curation_report.num_train} training and "
            f"{curation_report.num_validation} validation conversations."
        )

    print("\n" + "=" * 40)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Conversations before: {manifest['total_conversations_before']}")
    print(f"Duplicates removed:   {manifest['duplicates_removed']}")
    print(f"Conversations after:  {manifest['total_conversations_after']}")
    print(f"Duplicate percentage: {manifest['duplicate_percentage']:.2f}%")
    print("-" * 40)
    print(f"Canonical JSONL:      {canonical_path}")
    print(f"Deduplicated JSONL:   {dedup_path}")
    print(f"Manifest JSON:        {manifest_path}")
    print(f"Duplicate JSON Rep:   {rep_json_path}")
    print(f"Duplicate MD Rep:     {rep_md_path}")
    print(f"Statistics JSON:      {config['outputs']['statistics_json']}")
    print(f"Statistics MD:        {config['outputs']['statistics_md']}")
    print(f"Review Sample JSON:   {config['outputs']['review_sample_json']}")
    print(f"Review Sample MD:     {config['outputs']['review_sample_md']}")
    if curation_report is not None:
        print("-" * 40)
        print(f"Training JSONL:       {config['outputs']['train_jsonl']}")
        print(f"Validation JSONL:     {config['outputs']['validation_jsonl']}")
        print(f"Curation Report JSON: {config['outputs']['curation_report_json']}")
        print(f"Curation Report MD:   {config['outputs']['curation_report_md']}")
        print("-" * 40)
        print(f"Train conversations:  {curation_report.num_train}")
        print(f"Val conversations:    {curation_report.num_validation}")
        for source, count in sorted(curation_report.counts_after_sampling.items()):
            print(f"  {source}: {count}")
    print("=" * 40 + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="scripts.run_pipeline",
        description="Execute the end-to-end dataset pipeline orchestration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline_config.yaml"),
        help="Path to pipeline configuration YAML (default: configs/pipeline_config.yaml).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate registries and configurations, print execution plan, and exit.",
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

    # 2. Support dry-run mode
    if args.dry_run:
        try:
            verify_dry_run(config, dataset_registry)
            sys.exit(0)
        except PipelineConfigError as e:
            logger.error("Dry-Run Validation Failed: %s", e)
            sys.exit(1)
        except Exception as e:
            logger.error("Unexpected validation failure: %s", e)
            sys.exit(1)

    # 3. Execute pipeline
    try:
        execute_pipeline(config, dataset_registry)
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
