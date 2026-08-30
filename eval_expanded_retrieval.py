"""Evaluate Phase 9 original-plus-expanded retrieval followed by reranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from database import SessionLocal
from evaluation.dataset import DatasetValidationError, load_dataset
from evaluation.reporting import write_reports
from evaluation.runner import reranked_candidate_to_retrieved, run_evaluation
from expanded_retrieval import retrieve_expanded_hybrid_candidates
from embeddings import get_embedding
from query_expansion import DEFAULT_EXPANSION_MODEL, ExpansionUsage
from reranking import DEFAULT_RERANKER_MODEL, RerankerUsage, rerank_candidates


DEFAULT_DATASET = Path("evaluation/datasets/phase3.json")
DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/reports/phase9-expanded")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate safe query expansion over the Phase 8 pipeline."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--expansion", choices=("enabled", "disabled"), default="enabled"
    )
    parser.add_argument("--dense-candidate-k", type=int, default=20)
    parser.add_argument("--lexical-candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--relevance-cutoff", type=int, default=2)
    parser.add_argument("--expansion-model", default=DEFAULT_EXPANSION_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-batch-size", type=int, default=10)
    parser.add_argument("--preview-characters", type=int, default=240)
    return parser.parse_args()


def _candidate_diagnostics(case, candidates) -> dict[str, object]:
    expected = set(case.relevant_chunk_hashes)
    hashes = [candidate.chunk_content_hash for candidate in candidates]
    first_rank = next(
        (rank for rank, content_hash in enumerate(hashes, start=1) if content_hash in expected),
        None,
    )
    return {
        "count": len(hashes),
        "first_relevant_rank": first_rank if case.answerable else None,
        "hit": bool(expected & set(hashes)) if case.answerable else None,
        "recall": (
            len(expected & set(hashes)) / len(expected)
            if case.answerable and expected
            else None
        ),
        "chunk_hashes": hashes,
    }


def main() -> int:
    args = parse_args()
    database = None
    try:
        if args.rerank_k < 10:
            raise ValueError("rerank-k must be at least 10 for standard metrics")
        if args.final_k > args.rerank_k:
            raise ValueError("final-k cannot exceed rerank-k")
        dataset = load_dataset(args.dataset)
        database = SessionLocal()
        case_details: dict[str, dict[str, object]] = {}
        expansion_usage: list[ExpansionUsage] = []
        reranker_usage: list[RerankerUsage] = []

        def retrieve(case, _top_k):
            print(f"[{case.case_id}] retrieving candidates...", flush=True)
            retrieval = retrieve_expanded_hybrid_candidates(
                case.question,
                database,
                case.filename,
                expansion_enabled=args.expansion == "enabled",
                dense_candidate_k=args.dense_candidate_k,
                lexical_candidate_k=args.lexical_candidate_k,
                fused_top_k=args.rerank_k,
                rrf_k=args.rrf_k,
                document_content_hash=case.document_content_hash,
                expansion_model=args.expansion_model,
                embedding_function=lambda text: get_embedding(
                    text, timeout_seconds=30.0, max_retries=0
                ),
            )
            print(f"[{case.case_id}] reranking candidates...", flush=True)
            reranked = rerank_candidates(
                case.question,
                retrieval.fused_candidates,
                rerank_k=args.rerank_k,
                final_k=args.final_k,
                relevance_cutoff=args.relevance_cutoff,
                model=args.reranker_model,
                batch_size=args.reranker_batch_size,
            )
            expansion_usage.append(retrieval.expansion.usage)
            reranker_usage.append(reranked.usage)
            print(f"[{case.case_id}] complete", flush=True)
            case_details[case.case_id] = {
                "query_expansion": {
                    "original_query": retrieval.expansion.original_query,
                    "generated_query": retrieval.expansion.generated_query,
                    "expanded_query": retrieval.expansion.expanded_query,
                    "used_expansion": retrieval.expansion.used_expansion,
                    "protected_terms": list(retrieval.expansion.protected_terms),
                    "lexical_terms": list(retrieval.expansion.lexical_terms),
                    "fallback_reason": retrieval.expansion.fallback_reason,
                    "rationale": retrieval.expansion.rationale,
                    "latency_ms": retrieval.expansion.latency_ms,
                },
                "original_candidates": _candidate_diagnostics(
                    case, retrieval.original_result.fused_candidates
                ),
                "expanded_candidates": _candidate_diagnostics(
                    case, retrieval.fused_candidates
                ),
                "selected_count": len(reranked.selected_candidates),
                "selected_chunk_hashes": [
                    item.candidate.chunk_content_hash
                    for item in reranked.selected_candidates
                ],
                "reranker_used_fallback": reranked.used_fallback,
                "reranker_fallback_error": reranked.fallback_error,
                "reranker_latency_ms": reranked.latency_ms,
            }
            return [
                reranked_candidate_to_retrieved(item)
                for item in reranked.ranked_candidates
            ]

        report = run_evaluation(
            dataset,
            retrieve,
            top_k=args.rerank_k,
            preview_characters=args.preview_characters,
            configuration={
                "retrieval_method": "original_plus_expansion_hybrid_rrf_then_rerank",
                "expansion_enabled": args.expansion == "enabled",
                "expansion_model": args.expansion_model,
                "generated_alternatives": 1,
                "original_query_always_searched": True,
                "dense_candidate_k_per_query": args.dense_candidate_k,
                "lexical_candidate_k_per_query": args.lexical_candidate_k,
                "rrf_k": args.rrf_k,
                "candidate_limit": args.rerank_k,
                "rerank_k": args.rerank_k,
                "final_evidence_k": args.final_k,
                "relevance_cutoff": args.relevance_cutoff,
                "reranker_model": args.reranker_model,
                "reranker_batch_size": args.reranker_batch_size,
            },
        )
        for case in report["cases"]:
            case["query_expansion"] = case_details[case["id"]]["query_expansion"]
        report["query_expansion_summary"] = {
            "requests": sum(item.requests for item in expansion_usage),
            "input_tokens": sum(item.input_tokens for item in expansion_usage),
            "output_tokens": sum(item.output_tokens for item in expansion_usage),
            "estimated_cost_usd": sum(
                item.estimated_cost_usd for item in expansion_usage
            ),
            "used_cases": sum(
                bool(details["query_expansion"]["used_expansion"])
                for details in case_details.values()
            ),
            "fallback_cases": sum(
                not bool(details["query_expansion"]["used_expansion"])
                and args.expansion == "enabled"
                for details in case_details.values()
            ),
            "case_details": case_details,
        }
        report["reranker_summary"] = {
            "requests": sum(item.requests for item in reranker_usage),
            "input_tokens": sum(item.input_tokens for item in reranker_usage),
            "output_tokens": sum(item.output_tokens for item in reranker_usage),
            "estimated_cost_usd": sum(
                item.estimated_cost_usd for item in reranker_usage
            ),
            "fallback_cases": sum(
                bool(details["reranker_used_fallback"])
                for details in case_details.values()
            ),
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
    expansion_summary = report["query_expansion_summary"]
    reranker_summary = report["reranker_summary"]
    print(f"Run ID: {report['run']['run_id']}")
    print(f"Hit@5: {aggregate['hit_at_5']:.4f}")
    print(f"MRR: {aggregate['mrr']:.4f}")
    print(f"Expansion used: {expansion_summary['used_cases']}/{len(dataset.cases)}")
    print(f"Expansion fallbacks: {expansion_summary['fallback_cases']}")
    print(f"Reranker fallbacks: {reranker_summary['fallback_cases']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 1 if reranker_summary["fallback_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
