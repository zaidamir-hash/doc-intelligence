"""Command-line entry point for independent Phase 6 lexical evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from database import SessionLocal
from evaluation.dataset import DatasetValidationError, load_dataset
from evaluation.reporting import write_reports
from evaluation.runner import retrieve_lexical_chunks, run_evaluation
from lexical_retrieval import (
    DEFAULT_EXACT_MATCH_BOOST,
    RANK_NORMALIZATION,
    TEXT_SEARCH_CONFIGURATION,
)


DEFAULT_DATASET = Path("evaluation/datasets/phase3.json")
DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/reports/phase6-lexical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PostgreSQL lexical retrieval without embeddings."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--exact-matching",
        choices=("disabled", "supplement"),
        default="supplement",
    )
    parser.add_argument(
        "--exact-match-boost", type=float, default=DEFAULT_EXACT_MATCH_BOOST
    )
    parser.add_argument("--preview-characters", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database = None
    try:
        if args.top_k < 5:
            raise ValueError("top-k must be at least 5 for the standard metrics")
        if args.candidate_k < args.top_k:
            raise ValueError("candidate-k must be at least top-k")
        dataset = load_dataset(args.dataset)
        database = SessionLocal()
        cutoffs = (1, 3, 5, 10) if args.top_k >= 10 else (1, 3, 5)
        report = run_evaluation(
            dataset,
            lambda case, top_k: retrieve_lexical_chunks(
                case,
                database,
                top_k,
                candidate_k=args.candidate_k,
                exact_matching=args.exact_matching,
                exact_match_boost=args.exact_match_boost,
            ),
            top_k=args.top_k,
            cutoffs=cutoffs,
            preview_characters=args.preview_characters,
            configuration={
                "retrieval_method": "postgresql_lexical",
                "text_search_configuration": TEXT_SEARCH_CONFIGURATION,
                "query_parser": "OR-style websearch_to_tsquery",
                "rank_function": "ts_rank_cd",
                "rank_normalization": RANK_NORMALIZATION,
                "candidate_limit": args.candidate_k,
                "exact_matching": args.exact_matching,
                "exact_match_boost": args.exact_match_boost,
                "embedding_model": None,
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
        if database is not None:
            database.close()

    aggregate = report["aggregate_metrics"]
    print(f"Run ID: {report['run']['run_id']}")
    print(f"Hit@5: {aggregate['hit_at_5']:.4f}")
    print(f"MRR: {aggregate['mrr']:.4f}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
