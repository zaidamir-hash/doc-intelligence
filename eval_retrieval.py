"""Command-line entry point for the Phase 1 retrieval baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from database import SessionLocal
from evaluation.dataset import DatasetValidationError, load_dataset
from evaluation.reporting import write_reports
from evaluation.runner import retrieve_dense_chunks, run_evaluation


DEFAULT_DATASET = Path("evaluation/datasets/phase3.json")
DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current dense-only Lexis retriever without calling "
            "answer generation."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Versioned JSON dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Report directory (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of unique chunks to retain; must be at least 10",
    )
    parser.add_argument(
        "--preview-characters",
        type=int,
        default=240,
        help="Maximum characters stored in each diagnostic preview",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db = None
    try:
        dataset = load_dataset(args.dataset)
        db = SessionLocal()
        report = run_evaluation(
            dataset=dataset,
            retriever=lambda case, top_k: retrieve_dense_chunks(case, db, top_k),
            top_k=args.top_k,
            preview_characters=args.preview_characters,
        )
        json_path, markdown_path = write_reports(report, args.output_dir)
    except (DatasetValidationError, ValueError) as error:
        print(f"Evaluation configuration error: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    finally:
        if db is not None:
            db.close()

    aggregate = report["aggregate_metrics"]
    print(f"Run ID: {report['run']['run_id']}")
    print(f"Hit@5: {aggregate['hit_at_5']:.4f}")
    print(f"MRR: {aggregate['mrr']:.4f}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
