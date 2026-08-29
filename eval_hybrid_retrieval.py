"""Command-line entry point for Phase 7 hybrid RRF evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from database import SessionLocal
from evaluation.dataset import DatasetValidationError, load_dataset
from evaluation.reporting import write_reports
from evaluation.runner import retrieve_hybrid_chunks, run_evaluation


DEFAULT_DATASET = Path("evaluation/datasets/phase3.json")
DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/reports/phase7-hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate transparent dense + lexical Reciprocal Rank Fusion."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--dense-candidate-k", type=int, default=30)
    parser.add_argument("--lexical-candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--preview-characters", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database = None
    try:
        if args.top_k < 5:
            raise ValueError("top-k must be at least 5 for the standard metrics")
        if args.dense_candidate_k < args.top_k:
            raise ValueError("dense-candidate-k must be at least top-k")
        if args.lexical_candidate_k < args.top_k:
            raise ValueError("lexical-candidate-k must be at least top-k")
        if args.rrf_k <= 0:
            raise ValueError("rrf-k must be positive")
        dataset = load_dataset(args.dataset)
        database = SessionLocal()
        cutoffs = (1, 3, 5, 10) if args.top_k >= 10 else (1, 3, 5)
        report = run_evaluation(
            dataset,
            lambda case, top_k: retrieve_hybrid_chunks(
                case,
                database,
                top_k,
                dense_candidate_k=args.dense_candidate_k,
                lexical_candidate_k=args.lexical_candidate_k,
                rrf_k=args.rrf_k,
            ),
            top_k=args.top_k,
            cutoffs=cutoffs,
            preview_characters=args.preview_characters,
            configuration={
                "retrieval_method": "dense_lexical_rrf",
                "dense_candidate_limit": args.dense_candidate_k,
                "lexical_candidate_limit": args.lexical_candidate_k,
                "candidate_limit": args.dense_candidate_k
                + args.lexical_candidate_k,
                "rrf_k": args.rrf_k,
                "rrf_formula": "sum(1 / (rrf_k + source_rank))",
                "deduplication_identity": "document_id + chunk_content_hash",
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
