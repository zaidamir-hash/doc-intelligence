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
        help="Number of final chunks to retain (supported evaluation values: 5+)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
        help="Broad pgvector candidate-pool size (default: 30)",
    )
    parser.add_argument(
        "--metric",
        choices=("l2", "cosine"),
        default="cosine",
        help="pgvector distance operator (default: cosine)",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.8,
        help="Suppress candidates at or above this token-Jaccard similarity",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=None,
        help="Optional minimum interpreted dense similarity",
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
        if args.candidate_k < args.top_k:
            raise ValueError("candidate-k must be at least top-k")
        if args.top_k < 5:
            raise ValueError("top-k must be at least 5 for the standard metrics")
        cutoffs = (1, 3, 5, 10) if args.top_k >= 10 else (1, 3, 5)
        db = SessionLocal()
        report = run_evaluation(
            dataset=dataset,
            retriever=lambda case, top_k: retrieve_dense_chunks(
                case,
                db,
                top_k,
                candidate_k=args.candidate_k,
                metric=args.metric,
                relevance_threshold=args.relevance_threshold,
                duplicate_similarity_threshold=args.duplicate_threshold,
            ),
            top_k=args.top_k,
            cutoffs=cutoffs,
            preview_characters=args.preview_characters,
            configuration={
                "retrieval_method": "pgvector_dense_two_stage",
                "candidate_limit": args.candidate_k,
                "distance_metric": args.metric,
                "duplicate_similarity_threshold": args.duplicate_threshold,
                "relevance_threshold": args.relevance_threshold,
            },
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
