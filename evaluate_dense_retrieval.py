"""Run the reproducible Phase 5 dense-retrieval ablation study."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from database import SessionLocal
from dense_retrieval import (
    DenseCandidate,
    DenseSelection,
    expand_adjacent_context,
    retrieve_dense_candidates,
    select_dense_context,
)
from embeddings import EMBEDDING_MODEL, get_embeddings
from evaluation.dataset import EvaluationCase, EvaluationDataset, load_dataset
from evaluation.runner import RetrievedChunk, run_evaluation
from models import DOCUMENT_STATUS_READY, Document, DocumentChunk


DEFAULT_DATASET = Path("evaluation/datasets/phase3.json")
DEFAULT_OUTPUT = Path("evaluation/experiments/phase5_dense_comparison.json")
CALIBRATION_CASE_IDS = {
    "gov-ncpi-fy25-vs-fy24",
    "gov-policy-rate-reduction",
    "gov-climate-risk-fund-purpose",
    "gov-prism-plus-expansion",
    "gov-bitcoin-reserve-target-unanswerable",
}


@dataclass(frozen=True)
class Experiment:
    name: str
    metric: str = "cosine"
    candidate_k: int = 30
    top_k: int = 10
    duplicate_threshold: float = 0.8


EXPERIMENTS = (
    Experiment("l2-c30-k10", metric="l2"),
    Experiment("cosine-c10-k10", candidate_k=10),
    Experiment("cosine-c20-k10", candidate_k=20),
    Experiment("cosine-c30-k10"),
    Experiment("cosine-c60-k10", candidate_k=60),
    Experiment("cosine-c30-k5", top_k=5),
    Experiment("cosine-c30-k5-dedup90", top_k=5, duplicate_threshold=0.9),
    Experiment("cosine-c30-k10-dedup70", duplicate_threshold=0.7),
    Experiment("cosine-c30-k10-dedup90", duplicate_threshold=0.9),
    Experiment("cosine-c30-k10-no-near-dedup", duplicate_threshold=1.0),
)


def _subset(
    dataset: EvaluationDataset, case_ids: set[str], name: str, split: str
) -> EvaluationDataset:
    return replace(
        dataset,
        name=name,
        split=split,
        cases=tuple(case for case in dataset.cases if case.case_id in case_ids),
    )


def _as_retrieved(candidate: DenseCandidate) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=candidate.chunk_id,
        chunk_content_hash=candidate.chunk_content_hash,
        document_id=candidate.document_id,
        document_content_hash=candidate.document_content_hash,
        chunk_index=candidate.chunk_index,
        filename=candidate.filename,
        content=candidate.content,
        page_start=candidate.page_start,
        page_end=candidate.page_end,
        section_title=candidate.section_title,
        distance=candidate.distance,
        similarity=candidate.similarity,
        distance_metric=candidate.distance_metric,
        candidate_rank=candidate.dense_rank,
    )


def _metric_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "aggregate_metrics": report["aggregate_metrics"],
        "unanswerable_metrics": report["unanswerable_metrics"],
        "duration_ms": report["run"]["duration_ms"],
    }


def _run_database_experiment(
    experiment: Experiment,
    dataset: EvaluationDataset,
    db,
    query_embeddings: dict[str, list[float]],
) -> tuple[dict[str, object], dict[str, object]]:
    suppression = {"near_duplicate": 0, "below_threshold": 0}

    def retrieve(case: EvaluationCase, _top_k: int) -> list[RetrievedChunk]:
        candidates = retrieve_dense_candidates(
            case.question,
            db,
            case.filename,
            candidate_k=experiment.candidate_k,
            metric=experiment.metric,
            document_content_hash=case.document_content_hash,
            embedding_function=lambda _text: query_embeddings[case.case_id],
        )
        selection = select_dense_context(
            candidates,
            top_k=experiment.top_k,
            duplicate_similarity_threshold=experiment.duplicate_threshold,
        )
        for item in selection.suppressed:
            suppression[item.reason] += 1
        return [_as_retrieved(item) for item in selection.selected]

    cutoffs = (1, 3, 5) if experiment.top_k == 5 else (1, 3, 5, 10)
    report = run_evaluation(
        dataset,
        retrieve,
        top_k=experiment.top_k,
        cutoffs=cutoffs,
        configuration={
            "retrieval_method": "pgvector_exact_dense_two_stage",
            **asdict(experiment),
        },
    )
    return report, suppression


def _vector_norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _cosine_distance(first: Iterable[float], second: Iterable[float]) -> float:
    first_values = [float(value) for value in first]
    second_values = [float(value) for value in second]
    denominator = _vector_norm(first_values) * _vector_norm(second_values)
    if denominator == 0:
        return 1.0
    return 1.0 - sum(
        left * right for left, right in zip(first_values, second_values)
    ) / denominator


def _in_memory_retriever(
    chunks: list[DocumentChunk],
    embeddings: list[Iterable[float]],
    query_embeddings: dict[str, list[float]],
    document: Document,
):
    def retrieve(case: EvaluationCase, top_k: int) -> list[RetrievedChunk]:
        ranked = sorted(
            zip(chunks, embeddings),
            key=lambda pair: (
                _cosine_distance(query_embeddings[case.case_id], pair[1]),
                pair[0].chunk_index,
            ),
        )
        candidates = []
        for rank, (chunk, embedding) in enumerate(ranked[:30], start=1):
            distance = _cosine_distance(query_embeddings[case.case_id], embedding)
            candidates.append(
                DenseCandidate(
                    chunk_id=str(chunk.id),
                    chunk_content_hash=chunk.content_hash,
                    document_id=str(document.id),
                    document_content_hash=document.content_hash,
                    chunk_index=chunk.chunk_index,
                    filename=document.original_filename,
                    content=chunk.content,
                    passage_text=chunk.passage_text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    dense_rank=rank,
                    distance_metric="cosine",
                    distance=distance,
                    similarity=1.0 - distance,
                )
            )
        selection = select_dense_context(candidates, top_k=top_k)
        return [_as_retrieved(item) for item in selection.selected]

    return retrieve


def _adjacency_measurement(
    dataset: EvaluationDataset,
    db,
    query_embeddings: dict[str, list[float]],
) -> dict[str, object]:
    anchor_hits = 0
    expanded_hits = 0
    anchor_count = 0
    context_count = 0
    per_case = []
    for case in dataset.cases:
        if not case.answerable:
            continue
        candidates = retrieve_dense_candidates(
            case.question,
            db,
            case.filename,
            candidate_k=30,
            metric="cosine",
            document_content_hash=case.document_content_hash,
            embedding_function=lambda _text: query_embeddings[case.case_id],
        )
        selection = select_dense_context(candidates, top_k=5)
        contexts = expand_adjacent_context(selection, db, adjacent_window=1)
        expected = set(case.relevant_chunk_hashes)
        anchor_hit = bool(expected & {item.chunk_content_hash for item in selection.selected})
        expanded_hit = bool(expected & {item.chunk_content_hash for item in contexts})
        anchor_hits += int(anchor_hit)
        expanded_hits += int(expanded_hit)
        anchor_count += len(selection.selected)
        context_count += len(contexts)
        per_case.append(
            {
                "case_id": case.case_id,
                "anchor_hit": anchor_hit,
                "expanded_hit": expanded_hit,
                "anchor_chunks": len(selection.selected),
                "expanded_context_chunks": len(contexts),
            }
        )
    total = len(per_case)
    return {
        "window": 1,
        "anchor_hit_rate": anchor_hits / total,
        "expanded_hit_rate": expanded_hits / total,
        "additional_relevant_cases": expanded_hits - anchor_hits,
        "anchor_chunks": anchor_count,
        "expanded_context_chunks": context_count,
        "additional_context_chunks": context_count - anchor_count,
        "per_case": per_case,
    }


def _threshold_calibration(
    calibration: EvaluationDataset,
    held_out: EvaluationDataset,
    db,
    query_embeddings: dict[str, list[float]],
) -> dict[str, object]:
    def scores(dataset: EvaluationDataset) -> list[dict[str, object]]:
        values = []
        for case in dataset.cases:
            candidates = retrieve_dense_candidates(
                case.question,
                db,
                case.filename,
                candidate_k=30,
                metric="cosine",
                document_content_hash=case.document_content_hash,
                embedding_function=lambda _text: query_embeddings[case.case_id],
            )
            relevant_scores = [
                item.similarity
                for item in candidates
                if item.chunk_content_hash in set(case.relevant_chunk_hashes)
            ]
            values.append(
                {
                    "case_id": case.case_id,
                    "answerable": case.answerable,
                    "top_similarity": candidates[0].similarity if candidates else 0.0,
                    "best_relevant_similarity": max(relevant_scores, default=None),
                }
            )
        return values

    calibration_scores = scores(calibration)
    held_out_scores = scores(held_out)
    candidate_thresholds = sorted(
        {
            0.0,
            1.0,
            *(float(item["top_similarity"]) for item in calibration_scores),
            *(
                float(item["best_relevant_similarity"])
                for item in calibration_scores
                if item["best_relevant_similarity"] is not None
            ),
        }
    )
    candidates = []
    for threshold in candidate_thresholds:
        retained = sum(
            item["answerable"]
            and item["best_relevant_similarity"] is not None
            and float(item["best_relevant_similarity"]) >= threshold
            for item in calibration_scores
        )
        refused = sum(
            not item["answerable"] and float(item["top_similarity"]) < threshold
            for item in calibration_scores
        )
        accuracy = (retained + refused) / len(calibration_scores)
        candidates.append((accuracy, -threshold, threshold, retained, refused))
    _, _, selected, retained, refused = max(candidates)
    held_out_relevant_retained = sum(
        item["best_relevant_similarity"] is not None
        and float(item["best_relevant_similarity"]) >= selected
        for item in held_out_scores
    )
    return {
        "status": "provisional_not_enabled_in_production",
        "reason": (
            "Only one labelled unanswerable query exists; the threshold is a "
            "learning experiment, not enough evidence for a safe default."
        ),
        "selected_similarity_threshold": selected,
        "calibration_case_ids": [case.case_id for case in calibration.cases],
        "held_out_case_ids": [case.case_id for case in held_out.cases],
        "calibration_relevant_retained": retained,
        "calibration_unanswerable_refused": refused,
        "held_out_relevant_retained": held_out_relevant_retained,
        "held_out_answerable_total": len(held_out.cases),
        "calibration_scores": calibration_scores,
        "held_out_scores": held_out_scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    all_ids = {case.case_id for case in dataset.cases}
    if not CALIBRATION_CASE_IDS < all_ids:
        raise ValueError("Phase 5 calibration IDs do not match the dataset")
    calibration = _subset(
        dataset, CALIBRATION_CASE_IDS, "lexis-phase5-calibration", "development"
    )
    held_out = _subset(
        dataset,
        all_ids - CALIBRATION_CASE_IDS,
        "lexis-phase5-held-out",
        "test",
    )

    query_embeddings = dict(
        zip(
            [case.case_id for case in dataset.cases],
            get_embeddings([case.question for case in dataset.cases]),
        )
    )
    db = SessionLocal()
    try:
        document = (
            db.query(Document)
            .filter(
                Document.original_filename == dataset.cases[0].filename,
                Document.status == DOCUMENT_STATUS_READY,
            )
            .one()
        )
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
        stored_embeddings = [chunk.embedding for chunk in chunks]
        passage_embeddings = get_embeddings([chunk.passage_text for chunk in chunks])

        experiment_results = {}
        for experiment in EXPERIMENTS:
            calibration_report, calibration_suppression = _run_database_experiment(
                experiment, calibration, db, query_embeddings
            )
            held_out_report, held_out_suppression = _run_database_experiment(
                experiment, held_out, db, query_embeddings
            )
            experiment_results[experiment.name] = {
                "configuration": asdict(experiment),
                "calibration": _metric_summary(calibration_report),
                "held_out": _metric_summary(held_out_report),
                "suppression": {
                    "calibration": calibration_suppression,
                    "held_out": held_out_suppression,
                },
            }

        embedding_text_results = {}
        for name, embeddings in (
            ("heading_enriched_content", stored_embeddings),
            ("passage_text_only", passage_embeddings),
        ):
            retriever = _in_memory_retriever(
                chunks, embeddings, query_embeddings, document
            )
            embedding_text_results[name] = {
                "calibration": _metric_summary(
                    run_evaluation(calibration, retriever, top_k=10)
                ),
                "held_out": _metric_summary(
                    run_evaluation(held_out, retriever, top_k=10)
                ),
            }

        eligible_names = [
            name
            for name, item in experiment_results.items()
            if item["configuration"]["metric"] == "cosine"
            and item["configuration"]["candidate_k"]
            >= 3 * item["configuration"]["top_k"]
            and item["configuration"]["duplicate_threshold"] == 0.9
        ]
        selected_name = max(
            eligible_names,
            key=lambda name: (
                experiment_results[name]["calibration"]["aggregate_metrics"]["hit_at_5"],
                experiment_results[name]["calibration"]["aggregate_metrics"]["mrr"],
                experiment_results[name]["calibration"]["aggregate_metrics"].get("ndcg_at_5", 0),
                -experiment_results[name]["configuration"]["top_k"],
            ),
        )
        norms = [_vector_norm(vector) for vector in stored_embeddings]
        result = {
            "dataset": str(args.dataset),
            "embedding_model": EMBEDDING_MODEL,
            "selection_rule": (
                "Require cosine, a candidate pool at least three times final "
                "top-k, and conservative 0.90 deduplication; then use "
                "development Hit@5, MRR, and nDCG@5 and prefer less final context."
            ),
            "selected_dense_baseline": selected_name,
            "experiments": experiment_results,
            "embedding_normalization": {
                "count": len(norms),
                "minimum_norm": min(norms),
                "maximum_norm": max(norms),
                "mean_norm": statistics.mean(norms),
                "population_stddev": statistics.pstdev(norms),
            },
            "embedding_text_comparison": embedding_text_results,
            "adjacent_expansion": _adjacency_measurement(
                held_out, db, query_embeddings
            ),
            "threshold_calibration": _threshold_calibration(
                calibration, held_out, db, query_embeddings
            ),
        }
    finally:
        db.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected dense baseline: {result['selected_dense_baseline']}")
    print(f"Comparison report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
