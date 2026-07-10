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

## License

MIT
