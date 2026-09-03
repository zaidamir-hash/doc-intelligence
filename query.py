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
from settings import APP_SETTINGS


DEFAULT_DENSE_CANDIDATE_K = APP_SETTINGS.dense_candidate_k
DEFAULT_LEXICAL_CANDIDATE_K = APP_SETTINGS.lexical_candidate_k
DEFAULT_RRF_K = APP_SETTINGS.rrf_k
DEFAULT_RERANK_K = APP_SETTINGS.rerank_k
DEFAULT_FINAL_EVIDENCE_K = APP_SETTINGS.final_evidence_k
DEFAULT_RELEVANCE_CUTOFF = APP_SETTINGS.relevance_cutoff
DEFAULT_RERANKER_BATCH_SIZE = APP_SETTINGS.reranker_batch_size


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
            timeout_seconds=APP_SETTINGS.model_request_timeout_seconds,
            max_retries=APP_SETTINGS.model_max_retries,
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
    *,
    document_id: str | None = None,
    retrieval_mode: str = "hybrid_rrf_rerank_expansion",
    include_debug: bool = True,
    retrieval_configuration: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the normal answer contract and optional learning diagnostics."""

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
    relevance_cutoff = (
        retrieval_configuration.get("relevance_cutoff")
        if retrieval_configuration
        else DEFAULT_RELEVANCE_CUTOFF
    )
    retrieval_candidates = []
    for item in result.reranking.ranked_candidates:
        selected = (
            item.candidate.document_id,
            item.candidate.chunk_content_hash,
        ) in evidence_identities
        if selected:
            relevance_decision = "selected_evidence"
        elif item.reranker_score is None:
            relevance_decision = "unscored_fallback"
        elif item.reranker_score < relevance_cutoff:
            relevance_decision = "below_relevance_cutoff"
        else:
            relevance_decision = "eligible_not_selected"
        retrieval_candidates.append(
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
            "dense_rank": item.candidate.dense_rank,
            "dense_similarity": item.candidate.dense_similarity,
            "lexical_rank": item.candidate.lexical_rank,
            "lexical_score": item.candidate.lexical_score,
            "fused_rank": item.candidate.fused_rank,
            "fused_score": item.candidate.fused_score,
            "expanded_dense_rank": getattr(
                item.candidate, "expanded_dense_rank", None
            ),
            "expanded_lexical_rank": getattr(
                item.candidate, "expanded_lexical_rank", None
            ),
            "reranked_rank": item.final_rank,
            "reranker_score": item.reranker_score,
            "selected_for_generation": selected,
            "relevance_decision": relevance_decision,
        }
        )

    response: dict[str, object] = {
        "question": question,
        "document_id": document_id,
        "retrieval_mode": retrieval_mode,
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
    }
    if include_debug:
        expansion_fallback = result.retrieval.expansion.fallback_reason
        if expansion_fallback and expansion_fallback != "expansion disabled":
            expansion_fallback = (
                "Expansion was rejected or failed; the original query was used."
            )
        response["retrieval_candidates"] = retrieval_candidates
        response["retrieval_configuration"] = retrieval_configuration or {
            "active_mode": retrieval_mode,
            "relevance_cutoff": relevance_cutoff,
        }
        response["diagnostics"] = {
            "query_expansion": {
                "original_query": result.retrieval.expansion.original_query,
                "generated_query": result.retrieval.expansion.generated_query,
                "expanded_query": result.retrieval.expansion.expanded_query,
                "used_expansion": result.retrieval.expansion.used_expansion,
                "fallback_reason": expansion_fallback,
            },
            "reranker_used_fallback": result.reranking.used_fallback,
            "reranker_fallback_error": (
                "Reranking failed; fused order was retained."
                if result.reranking.fallback_error
                else None
            ),
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
            "generation_repair_attempted": result.answer.repair_attempted,
            "generation_repair_succeeded": result.answer.repair_succeeded,
            "generation_fallback_error": (
                "Answer generation failed; Lexis returned a grounded refusal."
                if result.answer.fallback_error
                else None
            ),
            "generation_repair_error": (
                "Answer repair failed; Lexis returned a grounded refusal."
                if result.answer.repair_error
                else None
            ),
            "rejected_claims": [
                {
                    "text": claim.text,
                    "source_ids": list(claim.source_ids),
                    "supporting_quotes": list(claim.supporting_quotes),
                }
                for claim in result.answer.rejected_claims
            ],
        }
    return response
