"""Transparent hybrid retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from dense_retrieval import DenseCandidate, retrieve_dense_candidates
from embeddings import get_embedding
from lexical_retrieval import LexicalCandidate, retrieve_lexical_candidates


EmbeddingFunction = Callable[[str], list[float]]


@dataclass(frozen=True)
class FusedCandidate:
    """One stable chunk with traceable dense, lexical, and RRF diagnostics."""

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
    fused_rank: int
    fused_score: float
    dense_rank: int | None
    dense_distance: float | None
    dense_similarity: float | None
    dense_rrf_contribution: float
    lexical_rank: int | None
    lexical_score: float | None
    fts_score: float | None
    exact_match_count: int
    exact_terms_matched: tuple[str, ...]
    parsed_search_text: str | None
    lexical_rrf_contribution: float
    source_count: int


@dataclass(frozen=True)
class HybridRetrievalResult:
    """Independent source lists plus their deduplicated fused ranking."""

    dense_candidates: tuple[DenseCandidate, ...]
    lexical_candidates: tuple[LexicalCandidate, ...]
    fused_candidates: tuple[FusedCandidate, ...]


def rrf_contribution(rank: int | None, rrf_k: int) -> float:
    """Calculate one retriever's `1 / (rrf_k + rank)` contribution."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if rank is None:
        return 0.0
    if rank <= 0:
        raise ValueError("rank must be positive")
    return 1.0 / (rrf_k + rank)


def _stable_identity(
    candidate: DenseCandidate | LexicalCandidate,
) -> tuple[str, str]:
    return candidate.document_id, candidate.chunk_content_hash


def fuse_candidates(
    dense_candidates: list[DenseCandidate],
    lexical_candidates: list[LexicalCandidate],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[FusedCandidate]:
    """Deduplicate by stable identity and fuse ranks without mixing raw scores."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when supplied")

    dense_by_identity = {
        _stable_identity(candidate): candidate for candidate in dense_candidates
    }
    lexical_by_identity = {
        _stable_identity(candidate): candidate for candidate in lexical_candidates
    }
    identities = set(dense_by_identity) | set(lexical_by_identity)
    rows = []
    for identity in identities:
        dense = dense_by_identity.get(identity)
        lexical = lexical_by_identity.get(identity)
        canonical = dense or lexical
        if canonical is None:  # Defensive; the union guarantees this cannot occur.
            continue

        dense_rank = dense.dense_rank if dense else None
        lexical_rank = lexical.lexical_rank if lexical else None
        dense_contribution = rrf_contribution(dense_rank, rrf_k)
        lexical_contribution = rrf_contribution(lexical_rank, rrf_k)
        fused_score = dense_contribution + lexical_contribution
        rows.append(
            {
                "canonical": canonical,
                "dense": dense,
                "lexical": lexical,
                "dense_rank": dense_rank,
                "lexical_rank": lexical_rank,
                "dense_contribution": dense_contribution,
                "lexical_contribution": lexical_contribution,
                "fused_score": fused_score,
                "source_count": int(dense is not None) + int(lexical is not None),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["fused_score"],
            -row["source_count"],
            min(
                rank
                for rank in (row["dense_rank"], row["lexical_rank"])
                if rank is not None
            ),
            row["canonical"].chunk_index,
            row["canonical"].chunk_id,
        )
    )
    if top_k is not None:
        rows = rows[:top_k]

    fused = []
    for fused_rank, row in enumerate(rows, start=1):
        canonical = row["canonical"]
        dense = row["dense"]
        lexical = row["lexical"]
        fused.append(
            FusedCandidate(
                chunk_id=canonical.chunk_id,
                chunk_content_hash=canonical.chunk_content_hash,
                document_id=canonical.document_id,
                document_content_hash=canonical.document_content_hash,
                chunk_index=canonical.chunk_index,
                filename=canonical.filename,
                content=canonical.content,
                passage_text=canonical.passage_text,
                page_start=canonical.page_start,
                page_end=canonical.page_end,
                section_title=canonical.section_title,
                fused_rank=fused_rank,
                fused_score=row["fused_score"],
                dense_rank=row["dense_rank"],
                dense_distance=dense.distance if dense else None,
                dense_similarity=dense.similarity if dense else None,
                dense_rrf_contribution=row["dense_contribution"],
                lexical_rank=row["lexical_rank"],
                lexical_score=lexical.lexical_score if lexical else None,
                fts_score=lexical.fts_score if lexical else None,
                exact_match_count=lexical.exact_match_count if lexical else 0,
                exact_terms_matched=(
                    lexical.exact_terms_matched if lexical else ()
                ),
                parsed_search_text=(lexical.parsed_search_text if lexical else None),
                lexical_rrf_contribution=row["lexical_contribution"],
                source_count=row["source_count"],
            )
        )
    return fused


def retrieve_hybrid_candidates(
    query: str,
    db: Session,
    filename: str,
    *,
    dense_candidate_k: int = 30,
    lexical_candidate_k: int = 30,
    fused_top_k: int | None = 30,
    rrf_k: int = 60,
    document_content_hash: str | None = None,
    embedding_function: EmbeddingFunction = get_embedding,
) -> HybridRetrievalResult:
    """Retrieve both source lists independently and fuse their rank positions."""

    dense_candidates = retrieve_dense_candidates(
        query,
        db,
        filename,
        candidate_k=dense_candidate_k,
        metric="cosine",
        document_content_hash=document_content_hash,
        embedding_function=embedding_function,
    )
    lexical_candidates = retrieve_lexical_candidates(
        query,
        db,
        filename,
        candidate_k=lexical_candidate_k,
        document_content_hash=document_content_hash,
        exact_matching="supplement",
    )
    fused_candidates = fuse_candidates(
        dense_candidates,
        lexical_candidates,
        rrf_k=rrf_k,
        top_k=fused_top_k,
    )
    return HybridRetrievalResult(
        dense_candidates=tuple(dense_candidates),
        lexical_candidates=tuple(lexical_candidates),
        fused_candidates=tuple(fused_candidates),
    )
