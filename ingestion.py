"""Transactional document ingestion and re-indexing lifecycle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from chunking import ChunkingConfig, StructuredChunk, chunk_extraction
from embeddings import EMBEDDING_MODEL, get_embedding, get_embeddings
from models import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_READY,
    DOCUMENT_STATUS_SUPERSEDED,
    Document,
    DocumentChunk,
)
from pdf_processing import EXTRACTION_VERSION, ExtractionResult, extract_pdf
from settings import APP_SETTINGS


EmbeddingFunction = Callable[[str], list[float]]


class NoExtractableTextError(ValueError):
    """Raised when a PDF produces no indexable chunks."""


class EmbeddingGenerationError(RuntimeError):
    """Raised when the configured embedding service cannot index a document."""


@dataclass(frozen=True)
class IngestionResult:
    document_id: UUID
    filename: str
    document_content_hash: str
    extraction: ExtractionResult
    chunks: tuple[StructuredChunk, ...]
    reused_existing_index: bool
    reindexed_existing_document: bool


def calculate_document_hash(contents: bytes) -> str:
    """Return the stable SHA-256 identity of the exact uploaded PDF bytes."""

    return hashlib.sha256(contents).hexdigest()


def chunking_config_metadata(config: ChunkingConfig) -> dict[str, int | str]:
    return {
        "max_tokens": config.max_tokens,
        "overlap_tokens": config.overlap_tokens,
        "min_chunk_tokens": config.min_chunk_tokens,
        "encoding_name": config.encoding_name,
    }


def _index_is_current(document: Document, config: ChunkingConfig) -> bool:
    return (
        document.status == DOCUMENT_STATUS_READY
        and document.extraction_version == EXTRACTION_VERSION
        and document.chunking_version == config.version
        and document.chunking_config == chunking_config_metadata(config)
        and document.embedding_model == EMBEDDING_MODEL
        and document.chunk_count > 0
    )


def _record_failed_processing(
    database: Session, document_id: UUID, message: str
) -> None:
    database.rollback()
    document = database.get(Document, document_id)
    if document is None or document.status != DOCUMENT_STATUS_PROCESSING:
        return
    document.status = DOCUMENT_STATUS_FAILED
    document.processing_error = message[:2000]
    database.commit()


def _prepare_processing_document(
    database: Session,
    filename: str,
    document_hash: str,
    config: ChunkingConfig,
) -> tuple[Document, bool]:
    document = (
        database.query(Document)
        .filter(
            Document.original_filename == filename,
            Document.content_hash == document_hash,
        )
        .one_or_none()
    )
    if document is None:
        document = Document(
            original_filename=filename,
            content_hash=document_hash,
            page_count=0,
            chunk_count=0,
            status=DOCUMENT_STATUS_PROCESSING,
            extraction_version=EXTRACTION_VERSION,
            chunking_version=config.version,
            chunking_config=chunking_config_metadata(config),
            embedding_model=EMBEDDING_MODEL,
        )
        database.add(document)
        database.commit()
        database.refresh(document)
        return document, False

    was_ready = document.status == DOCUMENT_STATUS_READY
    if not was_ready:
        document.status = DOCUMENT_STATUS_PROCESSING
        document.processing_error = None
        document.extraction_version = EXTRACTION_VERSION
        document.chunking_version = config.version
        document.chunking_config = chunking_config_metadata(config)
        document.embedding_model = EMBEDDING_MODEL
        database.commit()
        database.refresh(document)
    return document, was_ready


def ingest_document(
    database: Session,
    filename: str,
    contents: bytes,
    config: ChunkingConfig,
    *,
    embedding_function: EmbeddingFunction = get_embedding,
    force: bool = False,
) -> IngestionResult:
    """Create or refresh one document without exposing a partial ready index."""

    if not filename.strip():
        raise ValueError("filename cannot be empty")
    if not contents:
        raise ValueError("document contents cannot be empty")
    config.validate()
    document_hash = calculate_document_hash(contents)
    document, was_ready = _prepare_processing_document(
        database, filename, document_hash, config
    )

    try:
        extraction = extract_pdf(contents)
        generated_chunks = tuple(chunk_extraction(extraction, config))
        if not generated_chunks:
            raise NoExtractableTextError(
                "No extractable text found. PDF may be scanned or image-based."
            )

        if was_ready and _index_is_current(document, config) and not force:
            return IngestionResult(
                document_id=document.id,
                filename=filename,
                document_content_hash=document_hash,
                extraction=extraction,
                chunks=generated_chunks,
                reused_existing_index=True,
                reindexed_existing_document=False,
            )

        try:
            if embedding_function is get_embedding:
                embeddings = get_embeddings(
                    [chunk.content for chunk in generated_chunks],
                    timeout_seconds=APP_SETTINGS.model_request_timeout_seconds,
                    max_retries=APP_SETTINGS.model_max_retries,
                )
            else:
                embeddings = [
                    embedding_function(chunk.content) for chunk in generated_chunks
                ]
        except Exception as error:
            raise EmbeddingGenerationError(str(error)) from error
        if len(embeddings) != len(generated_chunks):
            raise EmbeddingGenerationError(
                "Embedding service returned a different number of vectors "
                "than the number of generated chunks."
            )

        previous_ready_documents = (
            database.query(Document)
            .filter(
                Document.original_filename == filename,
                Document.status == DOCUMENT_STATUS_READY,
                Document.id != document.id,
            )
            .all()
        )
        for previous in previous_ready_documents:
            previous.status = DOCUMENT_STATUS_SUPERSEDED
        database.flush()

        database.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id
        ).delete(synchronize_session=False)
        database.add_all(
            [
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk.content,
                    passage_text=chunk.passage_text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    content_hash=chunk.content_hash,
                    token_count=chunk.token_count,
                    overlap_token_count=chunk.overlap_token_count,
                    chunking_version=chunk.chunking_version,
                    embedding_model=EMBEDDING_MODEL,
                    boundary_kinds=list(chunk.boundary_kinds),
                    embedding=embedding,
                )
                for index, (chunk, embedding) in enumerate(
                    zip(generated_chunks, embeddings)
                )
            ]
        )
        document.page_count = extraction.page_count
        document.chunk_count = len(generated_chunks)
        document.status = DOCUMENT_STATUS_READY
        document.extraction_version = EXTRACTION_VERSION
        document.chunking_version = config.version
        document.chunking_config = chunking_config_metadata(config)
        document.embedding_model = EMBEDDING_MODEL
        document.processing_error = None
        database.commit()

        return IngestionResult(
            document_id=document.id,
            filename=filename,
            document_content_hash=document_hash,
            extraction=extraction,
            chunks=generated_chunks,
            reused_existing_index=False,
            reindexed_existing_document=was_ready,
        )
    except Exception as error:
        if not was_ready:
            _record_failed_processing(database, document.id, str(error))
        else:
            database.rollback()
        raise
