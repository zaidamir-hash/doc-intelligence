"""Central, environment-backed runtime settings for the Lexis application."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive number")
    return value


def _comma_separated(name: str, default: str) -> tuple[str, ...]:
    values = tuple(
        item.strip() for item in os.getenv(name, default).split(",") if item.strip()
    )
    if not values:
        raise RuntimeError(f"{name} must contain at least one value")
    return values


@dataclass(frozen=True)
class LexisSettings:
    """One explicit source for API, model, chunking, and retrieval settings."""

    api_key: str | None
    api_key_header: str
    cors_origins: tuple[str, ...]
    upload_rate_limit: str
    query_rate_limit: str
    max_upload_bytes: int
    max_filename_characters: int
    accepted_pdf_content_types: tuple[str, ...]
    embedding_model: str
    expansion_model: str
    reranker_model: str
    answer_model: str
    model_request_timeout_seconds: float
    model_max_retries: int
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    chunk_min_tokens: int
    chunk_encoding_name: str
    dense_candidate_k: int
    lexical_candidate_k: int
    rrf_k: int
    rerank_k: int
    final_evidence_k: int
    relevance_cutoff: int
    reranker_batch_size: int
    retrieval_mode: str

    @classmethod
    def from_environment(cls) -> "LexisSettings":
        settings = cls(
            api_key=os.getenv("API_KEY") or None,
            api_key_header=os.getenv("LEXIS_API_KEY_HEADER", "X-API-Key"),
            cors_origins=_comma_separated(
                "LEXIS_CORS_ORIGINS", "http://localhost:5173"
            ),
            upload_rate_limit=os.getenv(
                "LEXIS_UPLOAD_RATE_LIMIT", "5/minute"
            ),
            query_rate_limit=os.getenv("LEXIS_QUERY_RATE_LIMIT", "20/minute"),
            max_upload_bytes=_positive_int(
                "LEXIS_MAX_UPLOAD_BYTES", 25 * 1024 * 1024
            ),
            max_filename_characters=_positive_int(
                "LEXIS_MAX_FILENAME_CHARACTERS", 255
            ),
            accepted_pdf_content_types=_comma_separated(
                "LEXIS_ACCEPTED_PDF_CONTENT_TYPES",
                "application/pdf,application/x-pdf",
            ),
            embedding_model=os.getenv(
                "LEXIS_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            expansion_model=os.getenv(
                "LEXIS_EXPANSION_MODEL", "gpt-4o-mini-2024-07-18"
            ),
            reranker_model=os.getenv(
                "LEXIS_RERANKER_MODEL", "gpt-4o-mini-2024-07-18"
            ),
            answer_model=os.getenv(
                "LEXIS_ANSWER_MODEL", "gpt-4o-mini-2024-07-18"
            ),
            model_request_timeout_seconds=_positive_float(
                "LEXIS_MODEL_TIMEOUT_SECONDS", 30.0
            ),
            model_max_retries=_non_negative_int(
                "LEXIS_MODEL_MAX_RETRIES", 2
            ),
            chunk_max_tokens=_positive_int("LEXIS_CHUNK_MAX_TOKENS", 200),
            chunk_overlap_tokens=_non_negative_int(
                "LEXIS_CHUNK_OVERLAP_TOKENS", 30
            ),
            chunk_min_tokens=_positive_int("LEXIS_CHUNK_MIN_TOKENS", 40),
            chunk_encoding_name=os.getenv(
                "LEXIS_CHUNK_ENCODING", "cl100k_base"
            ),
            dense_candidate_k=_positive_int("LEXIS_DENSE_CANDIDATE_K", 20),
            lexical_candidate_k=_positive_int(
                "LEXIS_LEXICAL_CANDIDATE_K", 20
            ),
            rrf_k=_positive_int("LEXIS_RRF_K", 10),
            rerank_k=_positive_int("LEXIS_RERANK_K", 10),
            final_evidence_k=_positive_int("LEXIS_FINAL_EVIDENCE_K", 5),
            relevance_cutoff=_non_negative_int("LEXIS_RELEVANCE_CUTOFF", 2),
            reranker_batch_size=_positive_int(
                "LEXIS_RERANKER_BATCH_SIZE", 5
            ),
            retrieval_mode="hybrid_rrf_rerank_expansion",
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.api_key_header.strip():
            raise RuntimeError("LEXIS_API_KEY_HEADER cannot be empty")
        if not self.upload_rate_limit.strip() or not self.query_rate_limit.strip():
            raise RuntimeError("Lexis rate-limit settings cannot be empty")
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise RuntimeError(
                "LEXIS_CHUNK_OVERLAP_TOKENS must be smaller than "
                "LEXIS_CHUNK_MAX_TOKENS"
            )
        if self.chunk_min_tokens >= self.chunk_max_tokens:
            raise RuntimeError(
                "LEXIS_CHUNK_MIN_TOKENS must be smaller than "
                "LEXIS_CHUNK_MAX_TOKENS"
            )
        if self.final_evidence_k > self.rerank_k:
            raise RuntimeError(
                "LEXIS_FINAL_EVIDENCE_K cannot exceed LEXIS_RERANK_K"
            )
        if not 0 <= self.relevance_cutoff <= 3:
            raise RuntimeError("LEXIS_RELEVANCE_CUTOFF must be from 0 through 3")
        for name in (
            "embedding_model",
            "expansion_model",
            "reranker_model",
            "answer_model",
            "chunk_encoding_name",
        ):
            if not getattr(self, name).strip():
                raise RuntimeError(f"{name} cannot be empty")

    def public_configuration(self) -> dict[str, object]:
        """Expose learning/debug settings without secrets or database details."""

        return {
            "api": {
                "api_key_header": self.api_key_header,
                "upload_rate_limit": self.upload_rate_limit,
                "query_rate_limit": self.query_rate_limit,
                "max_upload_bytes": self.max_upload_bytes,
                "accepted_pdf_content_types": list(
                    self.accepted_pdf_content_types
                ),
                "scanned_pdf_behavior": (
                    "Text extraction only; scanned/image-only PDFs return 422 "
                    "because OCR is not enabled."
                ),
            },
            "models": {
                "embedding": self.embedding_model,
                "query_expansion": self.expansion_model,
                "reranker": self.reranker_model,
                "answer": self.answer_model,
                "request_timeout_seconds": self.model_request_timeout_seconds,
                "max_retries": self.model_max_retries,
            },
            "chunking": {
                "max_tokens": self.chunk_max_tokens,
                "overlap_tokens": self.chunk_overlap_tokens,
                "min_chunk_tokens": self.chunk_min_tokens,
                "encoding_name": self.chunk_encoding_name,
            },
            "retrieval": {
                "active_mode": self.retrieval_mode,
                "available_ablation_modes": [
                    "dense",
                    "lexical",
                    "hybrid_rrf",
                    "hybrid_rrf_rerank",
                    "hybrid_rrf_rerank_expansion",
                ],
                "pipeline_stages": [
                    "dense",
                    "lexical",
                    "hybrid_rrf",
                    "reranking",
                    "query_expansion",
                ],
                "dense_candidate_k": self.dense_candidate_k,
                "lexical_candidate_k": self.lexical_candidate_k,
                "rrf_k": self.rrf_k,
                "rerank_k": self.rerank_k,
                "final_evidence_k": self.final_evidence_k,
                "relevance_cutoff": self.relevance_cutoff,
                "reranker_batch_size": self.reranker_batch_size,
            },
        }


APP_SETTINGS = LexisSettings.from_environment()
