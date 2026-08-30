"""Multi-query hybrid retrieval built transparently on Phase 7 RRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from embeddings import get_embedding
from hybrid_retrieval import (
    FusedCandidate,
    HybridRetrievalResult,
    retrieve_hybrid_candidates,
    rrf_contribution,
)
from query_expansion import (
    DEFAULT_EXPANSION_MODEL,
    GenerateExpansionFunction,
    QueryExpansionResult,
    expand_query,
    generate_expansion_openai,
)


EmbeddingFunction = Callable[[str], list[float]]


@dataclass(frozen=True)
class ExpandedFusedCandidate(FusedCandidate):
    """A fused chunk with original-query and expanded-query rank traces."""

    expanded_dense_rank: int | None = None
    expanded_dense_similarity: float | None = None
    expanded_dense_rrf_contribution: float = 0.0
    expanded_lexical_rank: int | None = None
    expanded_lexical_score: float | None = None
    expanded_lexical_rrf_contribution: float = 0.0


@dataclass(frozen=True)
class ExpandedRetrievalResult:
    expansion: QueryExpansionResult
    original_result: HybridRetrievalResult
    expanded_result: HybridRetrievalResult | None
    fused_candidates: tuple[FusedCandidate, ...]


def _identity(candidate) -> tuple[str, str]:
    return candidate.document_id, candidate.chunk_content_hash


def fuse_query_results(
    original: HybridRetrievalResult,
    expanded: HybridRetrievalResult,
    *,
    rrf_k: int = 10,
    top_k: int | None = 10,
) -> list[ExpandedFusedCandidate]:
    """Fuse four independent source ranks with one contribution per signal."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when supplied")

    original_dense = {_identity(item): item for item in original.dense_candidates}
    original_lexical = {
        _identity(item): item for item in original.lexical_candidates
    }
    expanded_dense = {_identity(item): item for item in expanded.dense_candidates}
    expanded_lexical = {
        _identity(item): item for item in expanded.lexical_candidates
    }
    identities = (
        set(original_dense)
        | set(original_lexical)
        | set(expanded_dense)
        | set(expanded_lexical)
    )
    rows = []
    for identity in identities:
        dense = original_dense.get(identity)
        lexical = original_lexical.get(identity)
        alternative_dense = expanded_dense.get(identity)
        alternative_lexical = expanded_lexical.get(identity)
        canonical = dense or lexical or alternative_dense or alternative_lexical
        ranks = (
            dense.dense_rank if dense else None,
            lexical.lexical_rank if lexical else None,
            alternative_dense.dense_rank if alternative_dense else None,
            alternative_lexical.lexical_rank if alternative_lexical else None,
        )
        contributions = tuple(rrf_contribution(rank, rrf_k) for rank in ranks)
        rows.append(
            {
                "canonical": canonical,
                "dense": dense,
                "lexical": lexical,
                "alternative_dense": alternative_dense,
                "alternative_lexical": alternative_lexical,
                "ranks": ranks,
                "contributions": contributions,
                "score": sum(contributions),
                "source_count": sum(rank is not None for rank in ranks),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["score"],
            -row["source_count"],
            min(rank for rank in row["ranks"] if rank is not None),
            row["canonical"].chunk_index,
            row["canonical"].chunk_id,
        )
    )
    if top_k is not None:
        rows = rows[:top_k]

    fused = []
    for final_rank, row in enumerate(rows, start=1):
        canonical = row["canonical"]
        dense = row["dense"]
        lexical = row["lexical"]
        alternative_dense = row["alternative_dense"]
        alternative_lexical = row["alternative_lexical"]
        ranks = row["ranks"]
        contributions = row["contributions"]
        fused.append(
            ExpandedFusedCandidate(
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
                fused_rank=final_rank,
                fused_score=row["score"],
                dense_rank=ranks[0],
                dense_distance=dense.distance if dense else None,
                dense_similarity=dense.similarity if dense else None,
                dense_rrf_contribution=contributions[0],
                lexical_rank=ranks[1],
                lexical_score=lexical.lexical_score if lexical else None,
                fts_score=lexical.fts_score if lexical else None,
                exact_match_count=lexical.exact_match_count if lexical else 0,
                exact_terms_matched=(
                    lexical.exact_terms_matched if lexical else ()
                ),
                parsed_search_text=(lexical.parsed_search_text if lexical else None),
                lexical_rrf_contribution=contributions[1],
                source_count=row["source_count"],
                expanded_dense_rank=ranks[2],
                expanded_dense_similarity=(
                    alternative_dense.similarity if alternative_dense else None
                ),
                expanded_dense_rrf_contribution=contributions[2],
                expanded_lexical_rank=ranks[3],
                expanded_lexical_score=(
                    alternative_lexical.lexical_score
                    if alternative_lexical
                    else None
                ),
                expanded_lexical_rrf_contribution=contributions[3],
            )
        )
    return fused


def retrieve_expanded_hybrid_candidates(
    query: str,
    db: Session,
    filename: str,
    *,
    expansion_enabled: bool = True,
    dense_candidate_k: int = 20,
    lexical_candidate_k: int = 20,
    fused_top_k: int = 10,
    rrf_k: int = 10,
    document_content_hash: str | None = None,
    embedding_function: EmbeddingFunction = get_embedding,
    expansion_model: str = DEFAULT_EXPANSION_MODEL,
    expansion_generator: GenerateExpansionFunction = generate_expansion_openai,
) -> ExpandedRetrievalResult:
    """Always search the original query; add one validated expansion if available."""

    original = retrieve_hybrid_candidates(
        query,
        db,
        filename,
        dense_candidate_k=dense_candidate_k,
        lexical_candidate_k=lexical_candidate_k,
        fused_top_k=fused_top_k,
        rrf_k=rrf_k,
        document_content_hash=document_content_hash,
        embedding_function=embedding_function,
    )
    expansion = expand_query(
        query,
        enabled=expansion_enabled,
        model=expansion_model,
        generator=expansion_generator,
    )
    if not expansion.used_expansion or expansion.expanded_query is None:
        return ExpandedRetrievalResult(
            expansion=expansion,
            original_result=original,
            expanded_result=None,
            fused_candidates=original.fused_candidates,
        )

    alternative = retrieve_hybrid_candidates(
        expansion.expanded_query,
        db,
        filename,
        dense_candidate_k=dense_candidate_k,
        lexical_candidate_k=lexical_candidate_k,
        fused_top_k=fused_top_k,
        rrf_k=rrf_k,
        document_content_hash=document_content_hash,
        embedding_function=embedding_function,
    )
    fused = fuse_query_results(
        original,
        alternative,
        rrf_k=rrf_k,
        top_k=fused_top_k,
    )
    return ExpandedRetrievalResult(
        expansion=expansion,
        original_result=original,
        expanded_result=alternative,
        fused_candidates=tuple(fused),
    )
