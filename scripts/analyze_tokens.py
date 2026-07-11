"""CLI script to analyze the training dataset using the official Gemma tokenizer."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cbt_companion.models.token_statistics import TokenStatisticsReport

logger = logging.getLogger("cbt_companion.analyze_tokens")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.analyze_tokens",
        description="Analyze training dataset token lengths using Hugging Face AutoTokenizer.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/training/train.jsonl"),
        help="Path to the training JSONL dataset (default: outputs/training/train.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory to save output reports (default: outputs/reports).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-E2B",
        help="Hugging Face model tokenizer ID (default: google/gemma-4-E2B).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def calculate_percentile(sorted_data: list[int], percentile: float) -> int:
    """Calculate percentile using linear interpolation without NumPy."""
    if not sorted_data:
        return 0
    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return int(round(sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])))


def get_recommended_context_length(p99: int) -> int:
    """Determine the recommended model context size based on the 99th percentile."""
    if p99 <= 1024:
        return 1024
    elif p99 <= 2048:
        return 2048
    elif p99 <= 4096:
        return 4096
    elif p99 <= 8192:
        return 8192
    else:
        # Double from 8192
        length = 16384
        while length < p99:
            length *= 2
        return length


def get_bin_name(token_count: int) -> str:
    """Classify the token count into one of the histogram bins."""
    if token_count <= 128:
        return "0-128"
    elif token_count <= 256:
        return "129-256"
    elif token_count <= 512:
        return "257-512"
    elif token_count <= 1024:
        return "513-1024"
    elif token_count <= 2048:
        return "1025-2048"
    elif token_count <= 4096:
        return "2049-4096"
    elif token_count <= 8192:
        return "4097-8192"
    else:
        return "8193+"


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
    model_id: str = args.model_id

    if not input_path.exists():
        logger.error("Input training dataset not found: %s", input_path)
        sys.exit(1)

    # Load tokenizer
    logger.info("Loading tokenizer for model '%s'...", model_id)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if getattr(tokenizer, "chat_template", None) is None:
            # Fallback to official Gemma format supporting user/assistant roles
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "{{ '<start_of_turn>user\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
                "{% elif message['role'] == 'assistant' or message['role'] == 'model' %}"
                "{{ '<start_of_turn>model\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<start_of_turn>model\\n' }}"
                "{% endif %}"
            )
    except Exception as e:
        logger.error("Failed to load tokenizer for '%s': %s", model_id, e)
        sys.exit(1)

    logger.info("Analyzing token lengths in %s...", input_path)
    token_counts = []
    turn_counts = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON line: %s", e)
                sys.exit(1)

            messages = conv.get("messages", [])
            turn_counts.append(len(messages))

            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                tokens = tokenizer.encode(formatted)
                token_counts.append(len(tokens))
            except Exception as e:
                logger.error("Error tokenizing conversation: %s", e)
                sys.exit(1)

    total_conversations = len(token_counts)
    if total_conversations == 0:
        logger.error("No valid conversations parsed from input file.")
        sys.exit(1)

    # Sorted token counts for statistics
    sorted_tokens = sorted(token_counts)
    total_tokens = sum(sorted_tokens)
    average_tokens = round(total_tokens / total_conversations, 2)

    # Median tokens calculation
    mid = total_conversations // 2
    if total_conversations % 2 == 1:
        median_tokens = float(sorted_tokens[mid])
    else:
        median_tokens = (sorted_tokens[mid - 1] + sorted_tokens[mid]) / 2.0

    minimum_tokens = sorted_tokens[0]
    maximum_tokens = sorted_tokens[-1]

    # Percentiles
    p95 = calculate_percentile(sorted_tokens, 95.0)
    p99 = calculate_percentile(sorted_tokens, 99.0)

    # Average turns
    avg_turns = round(sum(turn_counts) / total_conversations, 2)

    # Recommended context
    recommended_context = get_recommended_context_length(p99)

    # Thresholds
    over_1024 = sum(1 for x in sorted_tokens if x > 1024)
    over_2048 = sum(1 for x in sorted_tokens if x > 2048)
    over_4096 = sum(1 for x in sorted_tokens if x > 4096)
    over_8192 = sum(1 for x in sorted_tokens if x > 8192)

    # Histogram categorisation
    histogram_bins = [
        "0-128",
        "129-256",
        "257-512",
        "513-1024",
        "1025-2048",
        "2049-4096",
        "4097-8192",
        "8193+",
    ]
    histogram = {bin_name: 0 for bin_name in histogram_bins}
    for count in sorted_tokens:
        bin_name = get_bin_name(count)
        histogram[bin_name] += 1

    # Instantiate Pydantic model
    report = TokenStatisticsReport(
        total_conversations=total_conversations,
        total_tokens=total_tokens,
        average_tokens=average_tokens,
        median_tokens=median_tokens,
        minimum_tokens=minimum_tokens,
        maximum_tokens=maximum_tokens,
        percentile_95_tokens=p95,
        percentile_99_tokens=p99,
        average_turns_per_conversation=avg_turns,
        recommended_context_length=recommended_context,
        conversations_over_1024=over_1024,
        conversations_over_2048=over_2048,
        conversations_over_4096=over_4096,
        conversations_over_8192=over_8192,
        histogram=histogram,
    )

    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write reports
    json_path = output_dir / "token_statistics.json"
    with json_path.open("w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Format Markdown output report
    md_lines = [
        "# Gemma Token Analysis Report",
        "",
        f"**Model ID**: `{model_id}`",
        "",
        "## Summary Metrics",
        f"- **Total conversations**: {report.total_conversations}",
        f"- **Total tokens**: {report.total_tokens}",
        f"- **Average tokens**: {report.average_tokens:.2f}",
        f"- **Median tokens**: {report.median_tokens:.1f}",
        f"- **Minimum tokens**: {report.minimum_tokens}",
        f"- **Maximum tokens**: {report.maximum_tokens}",
        f"- **95th percentile**: {report.percentile_95_tokens}",
        f"- **99th percentile**: {report.percentile_99_tokens}",
        f"- **Average turns per conversation**: {report.average_turns_per_conversation:.2f}",
        f"- **Recommended context length**: {report.recommended_context_length}",
        "",
        "## Conversations Over Length Thresholds",
        f"- **Tokens > 1024**: {report.conversations_over_1024}",
        f"- **Tokens > 2048**: {report.conversations_over_2048}",
        f"- **Tokens > 4096**: {report.conversations_over_4096}",
        f"- **Tokens > 8192**: {report.conversations_over_8192}",
        "",
        "## Token Length Histogram",
        "",
        "| Length Range | Conversation Count | Percentage |",
        "| :--- | :---: | :---: |",
    ]

    for bin_name in histogram_bins:
        count = histogram[bin_name]
        pct = round((count / total_conversations) * 100.0, 2)
        md_lines.append(f"| {bin_name} | {count} | {pct:.2f}% |")

    md_lines.append("")
    md_path = output_dir / "token_statistics.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Console output summary exactly matching layout requirements
    print(f"Total conversations: {report.total_conversations}")
    print(f"Total tokens: {report.total_tokens}")
    print(f"Average tokens: {report.average_tokens:.2f}")
    print(f"Median tokens: {report.median_tokens:.1f}")
    print(f"Minimum tokens: {report.minimum_tokens}")
    print(f"Maximum tokens: {report.maximum_tokens}")
    print(f"95th percentile: {report.percentile_95_tokens}")
    print(f"99th percentile: {report.percentile_99_tokens}")
    print(f"Average turns per conversation: {report.average_turns_per_conversation:.2f}")
    print(f"Recommended context length: {report.recommended_context_length}")
    print(f"Conversations > 1024: {report.conversations_over_1024}")
    print(f"Conversations > 2048: {report.conversations_over_2048}")
    print(f"Conversations > 4096: {report.conversations_over_4096}")
    print(f"Conversations > 8192: {report.conversations_over_8192}")


if __name__ == "__main__":
    main()
