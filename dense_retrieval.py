"""Explicit dense candidate retrieval and final context selection.

The two stages are intentionally separate:

1. PostgreSQL/pgvector retrieves a broad, ranked candidate pool.
2. Python selects a smaller context after thresholding and overlap-aware
   duplicate suppression.

Keeping these stages separate lets later hybrid retrieval reuse the candidate
pool without inheriting answer-context decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

from sqlalchemy.orm import Session

from embeddings import get_embedding
from models import DOCUMENT_STATUS_READY, Document, DocumentChunk


DistanceMetric = Literal["l2", "cosine"]
EmbeddingFunction = Callable[[str], list[float]]
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class DenseCandidate:
    """One pgvector result with its identity, provenance, rank, and scores."""

    chunk_id: str
    chunk_content_hash: str
    document_id: str
    document_content_hash: str
    chunk_index: int
    filename: str
    content: str
    passage_text: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    dense_rank: int
    distance_metric: DistanceMetric
    distance: float
    similarity: float


@dataclass(frozen=True)
class SuppressedCandidate:
    """A candidate excluded from final context and the inspectable reason."""

    candidate: DenseCandidate
    reason: Literal["below_threshold", "near_duplicate"]
    duplicate_of_chunk_id: str | None = None
    text_similarity: float | None = None


@dataclass(frozen=True)
class DenseSelection:
    """Final anchors plus all candidates deliberately excluded."""

    selected: tuple[DenseCandidate, ...]
    suppressed: tuple[SuppressedCandidate, ...]


@dataclass(frozen=True)
class ContextChunk:
    """An anchor or neighboring chunk included in generation context."""

    chunk_id: str
    chunk_content_hash: str
    document_id: str
    chunk_index: int
    content: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    source: Literal["dense", "adjacent"]
    anchor_chunk_id: str
    anchor_dense_rank: int


def validate_retrieval_sizes(candidate_k: int, top_k: int) -> None:
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")


def similarity_from_distance(distance: float, metric: DistanceMetric) -> float:
    """Convert a raw distance to a higher-is-better diagnostic similarity."""

    if metric == "cosine":
        return 1.0 - distance
    if metric == "l2":
        return 1.0 / (1.0 + distance)
    raise ValueError(f"Unsupported dense distance metric: {metric}")


def retrieve_dense_candidates(
    query: str,
    db: Session,
    filename: str,
    *,
    candidate_k: int = 30,
    metric: DistanceMetric = "cosine",
    document_content_hash: str | None = None,
    embedding_function: EmbeddingFunction = get_embedding,
) -> list[DenseCandidate]:
    """Retrieve a broad exact-search candidate pool from pgvector."""

    if not query.strip():
        raise ValueError("query cannot be empty")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if metric not in ("l2", "cosine"):
        raise ValueError(f"Unsupported dense distance metric: {metric}")

    query_embedding = embedding_function(query)
    if metric == "cosine":
        distance_expression = DocumentChunk.embedding.cosine_distance(query_embedding)
    else:
        distance_expression = DocumentChunk.embedding.l2_distance(query_embedding)

    filters = [
        Document.original_filename == filename,
        Document.status == DOCUMENT_STATUS_READY,
    ]
    if document_content_hash:
        filters.append(Document.content_hash == document_content_hash)

    rows = (
        db.query(DocumentChunk, Document, distance_expression.label("distance"))
        .join(Document)
        .filter(*filters)
        .order_by(distance_expression, DocumentChunk.chunk_index)
        .limit(candidate_k)
        .all()
    )
    return [
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
            distance_metric=metric,
            distance=float(distance),
            similarity=similarity_from_distance(float(distance), metric),
        )
        for rank, (chunk, document, distance) in enumerate(rows, start=1)
    ]


def token_jaccard(first: str, second: str) -> float:
    """Return set overlap for normalized words in two passages."""

    first_tokens = set(_WORD_PATTERN.findall(first.casefold()))
    second_tokens = set(_WORD_PATTERN.findall(second.casefold()))
    if not first_tokens and not second_tokens:
        return 1.0
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def select_dense_context(
    candidates: list[DenseCandidate],
    *,
    top_k: int = 8,
    relevance_threshold: float | None = None,
    duplicate_similarity_threshold: float = 0.8,
) -> DenseSelection:
    """Select final ranked anchors while recording every suppression decision."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= duplicate_similarity_threshold <= 1.0:
        raise ValueError("duplicate_similarity_threshold must be between 0 and 1")
    if relevance_threshold is not None and not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1")

    selected: list[DenseCandidate] = []
    suppressed: list[SuppressedCandidate] = []
    for candidate in candidates:
        if relevance_threshold is not None and candidate.similarity < relevance_threshold:
            suppressed.append(SuppressedCandidate(candidate, "below_threshold"))
            continue

        duplicate = None
        for kept in selected:
            text_similarity = token_jaccard(
                candidate.passage_text, kept.passage_text
            )
            if text_similarity >= duplicate_similarity_threshold:
                duplicate = (kept, text_similarity)
                break
        if duplicate is not None:
            kept, text_similarity = duplicate
            suppressed.append(
                SuppressedCandidate(
                    candidate,
                    "near_duplicate",
                    duplicate_of_chunk_id=kept.chunk_id,
                    text_similarity=text_similarity,
                )
            )
            continue

        selected.append(candidate)
        if len(selected) >= top_k:
            break

    return DenseSelection(tuple(selected), tuple(suppressed))


def expand_adjacent_context(
    selection: DenseSelection,
    db: Session,
    *,
    adjacent_window: int = 0,
) -> list[ContextChunk]:
    """Add ordered neighbor chunks around selected anchors without duplicates."""

    if adjacent_window < 0:
        raise ValueError("adjacent_window cannot be negative")
    contexts: dict[tuple[str, int], ContextChunk] = {}

    for anchor in selection.selected:
        lower = anchor.chunk_index - adjacent_window
        upper = anchor.chunk_index + adjacent_window
        neighbors = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == anchor.document_id,
                DocumentChunk.chunk_index.between(lower, upper),
            )
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
        for neighbor in neighbors:
            key = (str(neighbor.document_id), neighbor.chunk_index)
            is_anchor = str(neighbor.id) == anchor.chunk_id
            contexts.setdefault(
                key,
                ContextChunk(
                    chunk_id=str(neighbor.id),
                    chunk_content_hash=neighbor.content_hash,
                    document_id=str(neighbor.document_id),
                    chunk_index=neighbor.chunk_index,
                    content=neighbor.content,
                    page_start=neighbor.page_start,
                    page_end=neighbor.page_end,
                    section_title=neighbor.section_title,
                    source="dense" if is_anchor else "adjacent",
                    anchor_chunk_id=anchor.chunk_id,
                    anchor_dense_rank=anchor.dense_rank,
                ),
            )
    return sorted(
        contexts.values(),
        key=lambda item: (item.anchor_dense_rank, item.document_id, item.chunk_index),
    )
