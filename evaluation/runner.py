"""Run retrieval-only evaluation against the current Lexis dense search."""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy.orm import Session, joinedload

from embeddings import get_embedding
from models import DOCUMENT_STATUS_READY, Document, DocumentChunk

from .dataset import EvaluationCase, EvaluationDataset
from .metrics import DEFAULT_CUTOFFS, CaseMetrics, average_metrics, calculate_case_metrics


EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_METHOD = "pgvector_l2_exact_content_dedup"


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk plus the raw L2 distance used to rank it."""

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
    distance: float


EmbeddingFunction = Callable[[str], list[float]]
RetrieverFunction = Callable[[EvaluationCase, int], list[RetrievedChunk]]


def retrieve_dense_chunks(
    case: EvaluationCase,
    db: Session,
    top_k: int,
    embedding_function: EmbeddingFunction = get_embedding,
) -> list[RetrievedChunk]:
    """Mirror production dense retrieval while exposing L2 distances.

    Production retrieval requests `top_k * 3` rows ordered by L2 distance,
    removes exact duplicate content, and stops after `top_k` unique chunks.
    These baseline rules are deliberately unchanged.
    """

    query_embedding = embedding_function(case.question)
    distance_expression = DocumentChunk.embedding.l2_distance(query_embedding)
    document_filters = [
        Document.original_filename == case.filename,
        Document.status == DOCUMENT_STATUS_READY,
    ]
    if case.document_content_hash:
        document_filters.append(Document.content_hash == case.document_content_hash)
    rows = (
        db.query(DocumentChunk, distance_expression.label("distance"))
        .join(Document)
        .options(joinedload(DocumentChunk.document))
        .filter(*document_filters)
        .order_by(distance_expression)
        .limit(top_k * 3)
        .all()
    )

    seen_content: set[str] = set()
    unique_results: list[RetrievedChunk] = []
    for chunk, distance in rows:
        if chunk.content in seen_content:
            continue
        seen_content.add(chunk.content)
        unique_results.append(
            RetrievedChunk(
                chunk_id=str(chunk.id),
                chunk_content_hash=chunk.content_hash,
                document_id=str(chunk.document_id),
                document_content_hash=chunk.document.content_hash,
                chunk_index=chunk.chunk_index,
                filename=chunk.document.original_filename,
                content=chunk.content,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
                distance=float(distance),
            )
        )
        if len(unique_results) >= top_k:
            break
    return unique_results


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

    finished_at = datetime.now(timezone.utc)
    run_seed = (
        f"{dataset.name}|{started_at.isoformat()}|{top_k}|{RETRIEVAL_METHOD}"
    )
    run_id = hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:12]
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
        "configuration": {
            "retrieval_method": RETRIEVAL_METHOD,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": top_k,
            "candidate_limit": top_k * 3,
            "cutoffs": list(cutoffs),
            "preview_characters": preview_characters,
            "unanswerable_metric_policy": (
                "Exclude from Hit/MRR/Recall/nDCG until retrieval has a "
                "calibrated no-result threshold; retain diagnostics."
            ),
        },
        "aggregate_metrics": average_metrics(scored_metrics, cutoffs),
        "cases": diagnostics,
    }
