"""Orchestrate the production-shaped Lexis query pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from embeddings import get_embedding
from expanded_retrieval import (
    ExpandedRetrievalResult,
    retrieve_expanded_hybrid_candidates,
)
from grounded_generation import (
    DEFAULT_ANSWER_MODEL,
    GenerationContext,
    GroundedAnswer,
    build_generation_context,
    generate_grounded_answer,
)
from reranking import (
    DEFAULT_RERANKER_MODEL,
    RerankResult,
    rerank_candidates,
)


DEFAULT_DENSE_CANDIDATE_K = 20
DEFAULT_LEXICAL_CANDIDATE_K = 20
DEFAULT_RRF_K = 10
DEFAULT_RERANK_K = 10
DEFAULT_FINAL_EVIDENCE_K = 5
DEFAULT_RELEVANCE_CUTOFF = 2
DEFAULT_RERANKER_BATCH_SIZE = 5


@dataclass(frozen=True)
class GroundedQueryResult:
    """All observable stages from broad retrieval through grounded answer."""

    retrieval: ExpandedRetrievalResult
    reranking: RerankResult
    context: GenerationContext
    answer: GroundedAnswer


RetrievalFunction = Callable[..., ExpandedRetrievalResult]
RerankFunction = Callable[..., RerankResult]
AnswerFunction = Callable[..., GroundedAnswer]


def answer_document_question(
    question: str,
    db: Session,
    filename: str,
    *,
    document_content_hash: str | None = None,
    expansion_enabled: bool = True,
    dense_candidate_k: int = DEFAULT_DENSE_CANDIDATE_K,
    lexical_candidate_k: int = DEFAULT_LEXICAL_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
    rerank_k: int = DEFAULT_RERANK_K,
    final_evidence_k: int = DEFAULT_FINAL_EVIDENCE_K,
    relevance_cutoff: int = DEFAULT_RELEVANCE_CUTOFF,
    reranker_batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    retrieval_function: RetrievalFunction = retrieve_expanded_hybrid_candidates,
    rerank_function: RerankFunction = rerank_candidates,
    answer_function: AnswerFunction = generate_grounded_answer,
) -> GroundedQueryResult:
    """Run retrieval, evidence selection, context construction, and generation."""

    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question must not be empty")
    if not filename.strip():
        raise ValueError("filename must not be empty")

    retrieval = retrieval_function(
        cleaned_question,
        db,
        filename,
        expansion_enabled=expansion_enabled,
        dense_candidate_k=dense_candidate_k,
        lexical_candidate_k=lexical_candidate_k,
        fused_top_k=rerank_k,
        rrf_k=rrf_k,
        document_content_hash=document_content_hash,
        embedding_function=lambda text: get_embedding(
            text,
            timeout_seconds=30.0,
            max_retries=0,
        ),
    )
    reranking = rerank_function(
        cleaned_question,
        retrieval.fused_candidates,
        rerank_k=rerank_k,
        final_k=final_evidence_k,
        relevance_cutoff=relevance_cutoff,
        batch_size=reranker_batch_size,
        model=reranker_model,
    )
    context = build_generation_context(
        reranking.selected_candidates,
        relevance_cutoff=relevance_cutoff,
        max_evidence=final_evidence_k,
    )
    answer = answer_function(
        cleaned_question,
        context,
        model=answer_model,
    )
    return GroundedQueryResult(
        retrieval=retrieval,
        reranking=reranking,
        context=context,
        answer=answer,
    )


def build_query_response(
    question: str,
    result: GroundedQueryResult,
) -> dict[str, object]:
    """Expose broad candidates separately from evidence sent to generation."""

    cited_source_ids = {
        citation.source_id for citation in result.answer.citations
    }
    evidence = [
        {
            "source_id": item.source_id,
            "document_id": item.document_id,
            "document_content_hash": item.document_content_hash,
            "chunk_id": item.chunk_id,
            "chunk_content_hash": item.chunk_content_hash,
            "filename": item.filename,
            "chunk_index": item.chunk_index,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "section_title": item.section_title,
            "reranker_score": item.reranker_score,
            "reranked_rank": item.reranked_rank,
            "original_fused_rank": item.original_fused_rank,
            "cited": item.source_id in cited_source_ids,
            "preview": item.passage_text[:240],
        }
        for item in result.context.evidence
    ]
    evidence_identities = {
        (item.document_id, item.chunk_content_hash)
        for item in result.context.evidence
    }
    retrieval_candidates = [
        {
            "document_id": item.candidate.document_id,
            "document_content_hash": item.candidate.document_content_hash,
            "chunk_id": item.candidate.chunk_id,
            "chunk_content_hash": item.candidate.chunk_content_hash,
            "filename": item.candidate.filename,
            "chunk_index": item.candidate.chunk_index,
            "page_start": item.candidate.page_start,
            "page_end": item.candidate.page_end,
            "section_title": item.candidate.section_title,
            "fused_rank": item.candidate.fused_rank,
            "reranked_rank": item.final_rank,
            "reranker_score": item.reranker_score,
            "selected_for_generation": (
                item.candidate.document_id,
                item.candidate.chunk_content_hash,
            )
            in evidence_identities,
        }
        for item in result.reranking.ranked_candidates
    ]
    return {
        "question": question,
        "status": result.answer.status,
        "answer": result.answer.answer,
        "refusal_reason": result.answer.refusal_reason,
        "claims": [
            {
                "text": claim.text,
                "source_ids": list(claim.source_ids),
                "supporting_quotes": list(claim.supporting_quotes),
            }
            for claim in result.answer.claims
        ],
        "citations": [
            {
                "source_id": citation.source_id,
                "document_id": citation.document_id,
                "document_content_hash": citation.document_content_hash,
                "chunk_id": citation.chunk_id,
                "chunk_content_hash": citation.chunk_content_hash,
                "filename": citation.filename,
                "chunk_index": citation.chunk_index,
                "page_start": citation.page_start,
                "page_end": citation.page_end,
                "section_title": citation.section_title,
                "supported_claims": list(citation.supported_claims),
                "supporting_quotes": list(citation.supporting_quotes),
                "preview": citation.preview,
            }
            for citation in result.answer.citations
        ],
        "evidence": evidence,
        # Compatibility alias for the current frontend. These are only the
        # passages sent to generation, never every broad candidate.
        "sources": evidence,
        "retrieval_candidates": retrieval_candidates,
        "diagnostics": {
            "query_expansion": {
                "original_query": result.retrieval.expansion.original_query,
                "generated_query": result.retrieval.expansion.generated_query,
                "expanded_query": result.retrieval.expansion.expanded_query,
                "used_expansion": result.retrieval.expansion.used_expansion,
                "fallback_reason": result.retrieval.expansion.fallback_reason,
            },
            "reranker_used_fallback": result.reranking.used_fallback,
            "reranker_fallback_error": result.reranking.fallback_error,
            "suppressed_evidence": [
                {
                    "chunk_id": item.chunk_id,
                    "chunk_content_hash": item.chunk_content_hash,
                    "reason": item.reason,
                    "duplicate_of_chunk_id": item.duplicate_of_chunk_id,
                    "text_similarity": item.text_similarity,
                }
                for item in result.context.suppressed
            ],
            "generation_used_fallback": result.answer.used_fallback,
            "generation_fallback_error": result.answer.fallback_error,
            "rejected_claims": [
                {
                    "text": claim.text,
                    "source_ids": list(claim.source_ids),
                    "supporting_quotes": list(claim.supporting_quotes),
                }
                for claim in result.answer.rejected_claims
            ],
        },
    }
