"""Command-line entry point for the bounded Phase 11 answer evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from database import SessionLocal
from evaluation.answer_dataset import load_answer_dataset
from evaluation.answer_evaluation import (
    regrade_answer_report,
    run_answer_evaluation,
    write_answer_reports,
)


DEFAULT_DATASET = Path("evaluation/datasets/phase11_answers.json")
DEFAULT_JSON_OUTPUT = Path(
    "evaluation/experiments/phase11_answer_evaluation.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path(
    "evaluation/experiments/phase11_answer_evaluation.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate answer quality separately from retrieval using fixed gold "
            "facts, exact generation evidence, and deterministic rubrics."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
    )
    parser.add_argument(
        "--regrade-existing",
        type=Path,
        help=(
            "Reapply the deterministic grader to a saved JSON report without "
            "calling retrieval or hosted models."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_answer_dataset(args.dataset)
    if args.regrade_existing is not None:
        report = regrade_answer_report(
            json.loads(args.regrade_existing.read_text(encoding="utf-8")),
            dataset,
        )
    else:
        database = SessionLocal()
        try:
            report = run_answer_evaluation(dataset, database)
        finally:
            database.close()
    write_answer_reports(report, args.json_output, args.markdown_output)
    print(f"JSON report: {args.json_output}")
    print(f"Markdown report: {args.markdown_output}")
    print(
        "Retrieval success / answer success: "
        f"{report['aggregate']['retrieval_success_rate']:.4f} / "
        f"{report['aggregate']['answer_success_rate']:.4f}"
    )
    print(
        "Unsupported-claim rate: "
        f"{report['aggregate']['unsupported_claim_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
