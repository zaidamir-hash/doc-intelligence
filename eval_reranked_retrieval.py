"""Run Phase 8 hybrid retrieval followed by explicit passage reranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from database import SessionLocal
from evaluation.dataset import DatasetValidationError, load_dataset
from evaluation.reporting import write_reports
from evaluation.runner import reranked_candidate_to_retrieved, run_evaluation
from hybrid_retrieval import retrieve_hybrid_candidates
from reranking import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RERANKER_MODEL,
    RerankerUsage,
    rerank_candidates,
)


DEFAULT_DATASET = Path("evaluation/datasets/phase3.json")
DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/reports/phase8-reranked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 7 hybrid candidates after Phase 8 reranking."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--dense-candidate-k", type=int, default=20)
    parser.add_argument("--lexical-candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--relevance-cutoff", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--preview-characters", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database = None
    try:
        if args.rerank_k < 10:
            raise ValueError("rerank-k must be at least 10 for standard metrics")
        if args.final_k > args.rerank_k:
            raise ValueError("final-k cannot exceed rerank-k")
        if not 0 <= args.relevance_cutoff <= 3:
            raise ValueError("relevance-cutoff must be from 0 through 3")
        dataset = load_dataset(args.dataset)
        database = SessionLocal()
        case_details: dict[str, dict[str, object]] = {}
        usage_parts: list[RerankerUsage] = []

        def retrieve(case, _top_k):
            hybrid = retrieve_hybrid_candidates(
                case.question,
                database,
                case.filename,
                dense_candidate_k=args.dense_candidate_k,
                lexical_candidate_k=args.lexical_candidate_k,
                fused_top_k=args.rerank_k,
                rrf_k=args.rrf_k,
                document_content_hash=case.document_content_hash,
            )
            reranked = rerank_candidates(
                case.question,
                hybrid.fused_candidates,
                rerank_k=args.rerank_k,
                final_k=args.final_k,
                relevance_cutoff=args.relevance_cutoff,
                batch_size=args.batch_size,
                model=args.model,
            )
            usage_parts.append(reranked.usage)
            expected = set(case.relevant_chunk_hashes)
            candidate_hashes = {
                candidate.chunk_content_hash for candidate in hybrid.fused_candidates
            }
            case_details[case.case_id] = {
                "candidate_count": len(hybrid.fused_candidates),
                "candidate_hit": bool(expected & candidate_hashes)
                if case.answerable
                else None,
                "candidate_recall": len(expected & candidate_hashes) / len(expected)
                if case.answerable and expected
                else None,
                "selected_count": len(reranked.selected_candidates),
                "selected_chunk_hashes": [
                    item.candidate.chunk_content_hash
                    for item in reranked.selected_candidates
                ],
                "selected_scores": [
                    item.reranker_score for item in reranked.selected_candidates
                ],
                "used_fallback": reranked.used_fallback,
                "fallback_error": reranked.fallback_error,
                "reranker_latency_ms": reranked.latency_ms,
            }
            return [
                reranked_candidate_to_retrieved(candidate)
                for candidate in reranked.ranked_candidates
            ]

        report = run_evaluation(
            dataset,
            retrieve,
            top_k=args.rerank_k,
            configuration={
                "retrieval_method": "dense_lexical_rrf_then_pointwise_rerank",
                "dense_candidate_k": args.dense_candidate_k,
                "lexical_candidate_k": args.lexical_candidate_k,
                "rrf_k": args.rrf_k,
                "candidate_limit": args.rerank_k,
                "rerank_k": args.rerank_k,
                "final_evidence_k": args.final_k,
                "relevance_cutoff": args.relevance_cutoff,
                "reranker_model": args.model,
                "reranker_score_scale": "0=irrelevant, 1=topical, 2=supporting, 3=direct",
                "reranker_batch_size": args.batch_size,
                "reranker_request_timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
                "reranker_max_retries": 0,
            },
        )
        input_tokens = sum(usage.input_tokens for usage in usage_parts)
        output_tokens = sum(usage.output_tokens for usage in usage_parts)
        report["reranker_summary"] = {
            "requests": sum(usage.requests for usage in usage_parts),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": sum(
                usage.estimated_cost_usd for usage in usage_parts
            ),
            "fallback_cases": sum(
                bool(details["used_fallback"]) for details in case_details.values()
            ),
            "case_details": case_details,
        }
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
    summary = report["reranker_summary"]
    print(f"Run ID: {report['run']['run_id']}")
    print(f"Hit@5: {aggregate['hit_at_5']:.4f}")
    print(f"MRR: {aggregate['mrr']:.4f}")
    print(f"Reranker requests: {summary['requests']}")
    print(f"Estimated reranker cost: ${summary['estimated_cost_usd']:.6f}")
    print(f"Fallback cases: {summary['fallback_cases']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 1 if summary["fallback_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
