"""Run retrieval-only evaluation against the current Lexis dense search."""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy.orm import Session

from dense_retrieval import (
    DistanceMetric,
    retrieve_dense_candidates,
    select_dense_context,
)
from embeddings import get_embedding
from lexical_retrieval import LexicalCandidate, retrieve_lexical_candidates
from hybrid_retrieval import FusedCandidate, retrieve_hybrid_candidates
from reranking import RerankedCandidate

from .dataset import EvaluationCase, EvaluationDataset
from .metrics import DEFAULT_CUTOFFS, CaseMetrics, average_metrics, calculate_case_metrics


EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_METHOD = "pgvector_dense_two_stage"


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk plus its raw and interpreted dense scores."""

    chunk_id: str
    chunk_content_hash: str
    document_id: str
    document_content_hash: str
    chunk_index: int
    filename: str
    content: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    distance: float | None = None
    similarity: float | None = None
    distance_metric: str = "l2"
    candidate_rank: int | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fts_score: float | None = None
    exact_match_count: int = 0
    exact_terms_matched: tuple[str, ...] = ()
    parsed_search_text: str | None = None
    dense_rank: int | None = None
    fused_rank: int | None = None
    fused_score: float | None = None
    dense_rrf_contribution: float | None = None
    lexical_rrf_contribution: float | None = None
    source_count: int | None = None
    original_fused_rank: int | None = None
    reranker_score: int | None = None
    reranker_rationale: str | None = None
    reranker_used_fallback: bool = False
    expanded_dense_rank: int | None = None
    expanded_dense_similarity: float | None = None
    expanded_dense_rrf_contribution: float | None = None
    expanded_lexical_rank: int | None = None
    expanded_lexical_score: float | None = None
    expanded_lexical_rrf_contribution: float | None = None


EmbeddingFunction = Callable[[str], list[float]]
RetrieverFunction = Callable[[EvaluationCase, int], list[RetrievedChunk]]


def retrieve_dense_chunks(
    case: EvaluationCase,
    db: Session,
    top_k: int,
    embedding_function: EmbeddingFunction = get_embedding,
    *,
    candidate_k: int | None = None,
    metric: DistanceMetric = "cosine",
    relevance_threshold: float | None = None,
    duplicate_similarity_threshold: float = 0.8,
) -> list[RetrievedChunk]:
    """Run the shared Phase 5 candidate and context-selection stages."""

    resolved_candidate_k = candidate_k or max(top_k, top_k * 3)
    candidates = retrieve_dense_candidates(
        case.question,
        db,
        case.filename,
        candidate_k=resolved_candidate_k,
        metric=metric,
        document_content_hash=case.document_content_hash,
        embedding_function=embedding_function,
    )
    selection = select_dense_context(
        candidates,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
        duplicate_similarity_threshold=duplicate_similarity_threshold,
    )
    return [
        RetrievedChunk(
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
        for candidate in selection.selected
    ]


def _lexical_as_retrieved(candidate: LexicalCandidate) -> RetrievedChunk:
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
        lexical_rank=candidate.lexical_rank,
        lexical_score=candidate.lexical_score,
        fts_score=candidate.fts_score,
        exact_match_count=candidate.exact_match_count,
        exact_terms_matched=candidate.exact_terms_matched,
        parsed_search_text=candidate.parsed_search_text,
        candidate_rank=candidate.lexical_rank,
    )


def retrieve_lexical_chunks(
    case: EvaluationCase,
    db: Session,
    top_k: int,
    *,
    candidate_k: int | None = None,
    exact_matching: str = "supplement",
    exact_match_boost: float = 0.25,
) -> list[RetrievedChunk]:
    """Run lexical retrieval without invoking embeddings or vector search."""

    resolved_candidate_k = candidate_k or top_k
    candidates = retrieve_lexical_candidates(
        case.question,
        db,
        case.filename,
        candidate_k=resolved_candidate_k,
        document_content_hash=case.document_content_hash,
        exact_matching=exact_matching,
        exact_match_boost=exact_match_boost,
    )
    return [_lexical_as_retrieved(candidate) for candidate in candidates[:top_k]]


def hybrid_candidate_to_retrieved(candidate: FusedCandidate) -> RetrievedChunk:
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
        distance=candidate.dense_distance,
        similarity=candidate.dense_similarity,
        distance_metric="cosine",
        candidate_rank=candidate.fused_rank,
        dense_rank=candidate.dense_rank,
        lexical_rank=candidate.lexical_rank,
        lexical_score=candidate.lexical_score,
        fts_score=candidate.fts_score,
        exact_match_count=candidate.exact_match_count,
        exact_terms_matched=candidate.exact_terms_matched,
        parsed_search_text=candidate.parsed_search_text,
        fused_rank=candidate.fused_rank,
        fused_score=candidate.fused_score,
        dense_rrf_contribution=candidate.dense_rrf_contribution,
        lexical_rrf_contribution=candidate.lexical_rrf_contribution,
        source_count=candidate.source_count,
        expanded_dense_rank=getattr(candidate, "expanded_dense_rank", None),
        expanded_dense_similarity=getattr(
            candidate, "expanded_dense_similarity", None
        ),
        expanded_dense_rrf_contribution=getattr(
            candidate, "expanded_dense_rrf_contribution", None
        ),
        expanded_lexical_rank=getattr(candidate, "expanded_lexical_rank", None),
        expanded_lexical_score=getattr(
            candidate, "expanded_lexical_score", None
        ),
        expanded_lexical_rrf_contribution=getattr(
            candidate, "expanded_lexical_rrf_contribution", None
        ),
    )


def retrieve_hybrid_chunks(
    case: EvaluationCase,
    db: Session,
    top_k: int,
    embedding_function: EmbeddingFunction = get_embedding,
    *,
    dense_candidate_k: int = 30,
    lexical_candidate_k: int = 30,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Run both independent retrievers and return the top fused ranking."""

    result = retrieve_hybrid_candidates(
        case.question,
        db,
        case.filename,
        dense_candidate_k=dense_candidate_k,
        lexical_candidate_k=lexical_candidate_k,
        fused_top_k=top_k,
        rrf_k=rrf_k,
        document_content_hash=case.document_content_hash,
        embedding_function=embedding_function,
    )
    return [
        hybrid_candidate_to_retrieved(candidate)
        for candidate in result.fused_candidates
    ]


def reranked_candidate_to_retrieved(candidate: RerankedCandidate) -> RetrievedChunk:
    """Preserve Phase 7 diagnostics while adding the Phase 8 ranking trace."""

    retrieved = hybrid_candidate_to_retrieved(candidate.candidate)
    return RetrievedChunk(
        **{
            **asdict(retrieved),
            "candidate_rank": candidate.final_rank,
            "original_fused_rank": candidate.original_fused_rank,
            "reranker_score": candidate.reranker_score,
            "reranker_rationale": candidate.reranker_rationale,
            "reranker_used_fallback": candidate.used_fallback,
        }
    )


def _case_result(
    case: EvaluationCase,
    retrieved: list[RetrievedChunk],
    cutoffs: tuple[int, ...],
    preview_characters: int,
    elapsed_ms: float,
) -> tuple[dict[str, object], CaseMetrics | None]:
    if case.relevant_chunk_hashes:
        retrieved_labels = [chunk.chunk_content_hash for chunk in retrieved]
        relevance_labels = {
            content_hash: case.relevance_grades[chunk_index]
            for chunk_index, content_hash in zip(
                case.relevant_chunk_indices, case.relevant_chunk_hashes
            )
        }
        identity_mode = "stable_content_hash"
    else:
        retrieved_labels = [chunk.chunk_index for chunk in retrieved]
        relevance_labels = case.relevance_grades
        identity_mode = "legacy_chunk_index"
    metrics = (
        calculate_case_metrics(retrieved_labels, relevance_labels, cutoffs)
        if case.answerable
        else None
    )
    diagnostic: dict[str, object] = {
        "id": case.case_id,
        "filename": case.filename,
        "question": case.question,
        "answerable": case.answerable,
        "expected_chunk_indices": list(case.relevant_chunk_indices),
        "expected_chunk_hashes": list(case.relevant_chunk_hashes),
        "expected_document_content_hash": case.document_content_hash,
        "evaluation_identity_mode": identity_mode,
        "relevance_grades": {
            str(index): grade for index, grade in case.relevance_grades.items()
        },
        "reference_answer": case.reference_answer,
        "notes": case.notes,
        "tags": list(case.tags),
        "elapsed_ms": round(elapsed_ms, 3),
        "retrieved": [
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "chunk_content_hash": chunk.chunk_content_hash,
                "document_id": chunk.document_id,
                "document_content_hash": chunk.document_content_hash,
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_title": chunk.section_title,
                "distance": chunk.distance,
                "similarity": chunk.similarity,
                "distance_metric": chunk.distance_metric,
                "candidate_rank": chunk.candidate_rank,
                "lexical_rank": chunk.lexical_rank,
                "lexical_score": chunk.lexical_score,
                "fts_score": chunk.fts_score,
                "exact_match_count": chunk.exact_match_count,
                "exact_terms_matched": list(chunk.exact_terms_matched),
                "parsed_search_text": chunk.parsed_search_text,
                "dense_rank": chunk.dense_rank,
                "fused_rank": chunk.fused_rank,
                "fused_score": chunk.fused_score,
                "dense_rrf_contribution": chunk.dense_rrf_contribution,
                "lexical_rrf_contribution": chunk.lexical_rrf_contribution,
                "source_count": chunk.source_count,
                "original_fused_rank": chunk.original_fused_rank,
                "reranker_score": chunk.reranker_score,
                "reranker_rationale": chunk.reranker_rationale,
                "reranker_used_fallback": chunk.reranker_used_fallback,
                "expanded_dense_rank": chunk.expanded_dense_rank,
                "expanded_dense_similarity": chunk.expanded_dense_similarity,
                "expanded_dense_rrf_contribution": chunk.expanded_dense_rrf_contribution,
                "expanded_lexical_rank": chunk.expanded_lexical_rank,
                "expanded_lexical_score": chunk.expanded_lexical_score,
                "expanded_lexical_rrf_contribution": chunk.expanded_lexical_rrf_contribution,
                "preview": chunk.content[:preview_characters],
            }
            for rank, chunk in enumerate(retrieved, start=1)
        ],
        "metrics": None if metrics is None else asdict(metrics),
        "metric_status": (
            "scored"
            if case.answerable
            else "excluded_from_relevance_metrics_until_a_relevance_threshold_exists"
        ),
    }
    return diagnostic, metrics


def run_evaluation(
    dataset: EvaluationDataset,
    retriever: RetrieverFunction,
    top_k: int = 10,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    preview_characters: int = 240,
    configuration: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run every case and return a serializable report."""

    if top_k < max(cutoffs):
        raise ValueError("top_k must be at least the largest metric cutoff")
    if preview_characters <= 0:
        raise ValueError("preview_characters must be positive")
    if not dataset.cases:
        raise ValueError(
            "The dataset contains no cases. Add human-labelled cases before "
            "claiming or reproducing metrics."
        )

    started_at = datetime.now(timezone.utc)
    diagnostics: list[dict[str, object]] = []
    scored_metrics: list[CaseMetrics] = []
    unanswerable_empty_results = 0

    for case in dataset.cases:
        case_started = time.perf_counter()
        retrieved = retriever(case, top_k)
        elapsed_ms = (time.perf_counter() - case_started) * 1000
        diagnostic, metrics = _case_result(
            case, retrieved, cutoffs, preview_characters, elapsed_ms
        )
        diagnostics.append(diagnostic)
        if metrics is not None:
            scored_metrics.append(metrics)
        elif not retrieved:
            unanswerable_empty_results += 1

    finished_at = datetime.now(timezone.utc)
    run_seed = (
        f"{dataset.name}|{started_at.isoformat()}|{top_k}|{RETRIEVAL_METHOD}"
    )
    run_id = hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:12]
    report_configuration: dict[str, object] = {
        "retrieval_method": RETRIEVAL_METHOD,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": top_k,
        "candidate_limit": top_k,
        "cutoffs": list(cutoffs),
        "preview_characters": preview_characters,
        "unanswerable_metric_policy": (
            "Report empty-result rate separately from answerable ranking metrics."
        ),
    }
    if configuration:
        report_configuration.update(configuration)

    unanswerable_count = len(dataset.cases) - len(scored_metrics)
    return {
        "run": {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": round((finished_at - started_at).total_seconds() * 1000, 3),
            "python_version": platform.python_version(),
        },
        "dataset": {
            "schema_version": dataset.schema_version,
            "name": dataset.name,
            "description": dataset.description,
            "split": dataset.split,
            "total_cases": len(dataset.cases),
            "answerable_cases": len(scored_metrics),
            "unanswerable_cases": len(dataset.cases) - len(scored_metrics),
        },
        "configuration": report_configuration,
        "unanswerable_metrics": {
            "empty_result_rate": (
                unanswerable_empty_results / unanswerable_count
                if unanswerable_count
                else None
            ),
            "empty_results": unanswerable_empty_results,
            "total": unanswerable_count,
        },
        "aggregate_metrics": average_metrics(scored_metrics, cutoffs),
        "cases": diagnostics,
    }
