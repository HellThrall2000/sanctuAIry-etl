# CBT Companion

Dataset engineering pipeline for fine-tuning **Gemma 4 E2B** using **Unsloth**.

## Overview

CBT Companion is a modular, configuration-driven pipeline for curating, cleaning, normalizing, deduplicating, and exporting high-quality datasets for LLM fine-tuning. The architecture is designed around SOLID principles to support long-term maintainability and iterative dataset refinement.

## Project Structure

```
cbt-companion/
│
├── configs/              # YAML/JSON pipeline configurations
├── datasets/
│   ├── raw/              # Original downloaded datasets
│   ├── processed/        # Cleaned and transformed datasets
│   ├── merged/           # Combined multi-source datasets
│   └── final/            # Export-ready training datasets
│
├── scripts/
│   ├── download/         # Dataset acquisition scripts
│   ├── inspect/          # Data exploration and profiling
│   ├── preprocess/       # Cleaning and formatting
│   ├── normalize/        # Schema and format normalization
│   ├── merge/            # Multi-source dataset merging
│   ├── validate/         # Quality checks and assertions
│   ├── deduplicate/      # Near-duplicate and exact-duplicate removal
│   └── export/           # Final export and packaging
│
├── src/
│   └── cbt_companion/
│       ├── core/         # Base classes, protocols, and abstractions
│       ├── models/       # Data models and schemas
│       ├── pipeline/     # Pipeline orchestration
│       ├── loaders/      # Dataset loaders and adapters
│       ├── curation/     # Training-set cleaning, filtering and export
│       └── utils/        # Shared utilities
│
├── prompts/              # Prompt templates and system instructions
├── training/             # Training configs and launch scripts
├── evaluation/           # Evaluation scripts and benchmarks
├── outputs/              # Generated artifacts and results
├── logs/                 # Pipeline execution logs
└── tests/                # Unit and integration tests
```

## Requirements

- Python ≥ 3.12

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Design Principles

- **Configuration-driven**: Pipeline behavior is controlled by external config files, not hardcoded values.
- **Modular stages**: Each pipeline stage (download, preprocess, normalize, merge, validate, deduplicate, export) is an independent, composable unit.
- **SOLID architecture**: Abstractions and protocols decouple stages from concrete implementations.
- **Pathlib-first**: All file system operations use `pathlib.Path`.
- **Type-safe**: Strict type hints on all public interfaces.

## Running the Pipeline

A single command runs every stage and produces the training splits:

```bash
python scripts/run_pipeline.py
```

Add `--dry-run` to validate the configuration and print the execution plan without
downloading anything, or `-v` for debug logging. All behaviour is driven by
`configs/pipeline_config.yaml`.

Stages: **load → parse → normalize → validate → filter → merge → export → deduplicate
→ statistics → curate**.

## Canonical Dataset

The frozen canonical dataset is written to:

```
outputs/final/canonical_dataset.jsonl
```

This file contains one canonical `Conversation` JSON object per line (JSONL format, UTF-8 encoded). It preserves the full Pydantic schema including all metadata fields. This is the single source of truth for all downstream offline curation, quality review, and model training.

## Training Dataset

The curation stage turns the deduplicated corpus into fine-tuning splits:

```
outputs/training/train.jsonl
outputs/training/validation.jsonl
```

Each line is a chat record — `{"messages": [{"role": ..., "content": ...}, ...]}` —
optionally led by the system turn in `prompts/cbt_system_prompt.txt`. The record stops
short of a rendered prompt string on purpose: the chat template is applied by the
training notebook using the target model's own tokenizer, so the training format cannot
drift out of sync with what the model sees at inference.

Curation runs four stages, all configured under `curation:` in the pipeline config:

| Stage | Purpose |
| :--- | :--- |
| **Clean** | Repairs corpus artifacts — `_comma_` placeholders, HTML entities, missing spaces after sentence punctuation, bare URLs. The normalizers copy source text verbatim, so without this step these artifacts reach the model and get reproduced in its output. |
| **Quality** | Drops conversations that teach bad behaviour: assistant turns that echo the user, assistant turns too short to be a real response, and conversations not ending on an assistant turn. |
| **Rebalance** | Applies per-source caps. The merged corpus is ~86% EmpatheticDialogues, whose turns average ~36 characters of casual peer chat; uncapped it dominates the gradient and teaches short, generic replies. Within a capped source the most substantive conversations are kept. |
| **Split** | Deterministic shuffle and train/validation split under a fixed seed. |

`outputs/reports/curation_report.md` records the per-source funnel, every text repair
applied, and the reason each conversation was dropped.

To retune the mix, edit `curation.source_caps` — no code change is required:

```yaml
curation:
  source_caps:
    facebook/empathetic_dialogues: 2500   # cap this source
    thu-coai/esconv: null                 # keep every conversation
```

Set `curation.enabled: false` to stop after the deduplicated dataset.

## License

MIT
