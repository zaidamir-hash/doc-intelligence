"""Persistent document and chunk models for Lexis."""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import relationship

from database import Base


DOCUMENT_STATUS_PROCESSING = "processing"
DOCUMENT_STATUS_READY = "ready"
DOCUMENT_STATUS_FAILED = "failed"
DOCUMENT_STATUS_SUPERSEDED = "superseded"
DOCUMENT_STATUSES = (
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_READY,
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_SUPERSEDED,
)


class Document(Base):
    """One uploaded PDF version with a stable UUID and lifecycle state."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "original_filename", "content_hash", name="uq_documents_filename_hash"
        ),
        CheckConstraint("page_count >= 0", name="ck_documents_page_count"),
        CheckConstraint("chunk_count >= 0", name="ck_documents_chunk_count"),
        CheckConstraint(
            "status IN ('processing', 'ready', 'failed', 'superseded')",
            name="ck_documents_status",
        ),
        CheckConstraint(
            "jsonb_typeof(chunking_config) = 'object'",
            name="ck_documents_chunking_config",
        ),
        Index("ix_documents_status", "status"),
        Index(
            "uq_documents_ready_filename",
            "original_filename",
            unique=True,
            postgresql_where=text("status = 'ready'"),
        ),
    )

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    original_filename = Column(String, nullable=False)
    content_hash = Column(String(64), nullable=False)
    page_count = Column(Integer, nullable=False, default=0, server_default="0")
    chunk_count = Column(Integer, nullable=False, default=0, server_default="0")
    status = Column(String(20), nullable=False)
    extraction_version = Column(String, nullable=False)
    chunking_version = Column(String, nullable=False)
    chunking_config = Column(JSONB, nullable=False, default=dict, server_default="{}")
    embedding_model = Column(String, nullable=False)
    processing_error = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentChunk.chunk_index",
    )


class DocumentChunk(Base):
    """One embedded passage with stable provenance and lexical representation."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_index", name="uq_chunks_document_index"
        ),
        UniqueConstraint(
            "document_id", "content_hash", name="uq_chunks_document_hash"
        ),
        CheckConstraint("chunk_index >= 0", name="ck_chunks_index"),
        CheckConstraint(
            "page_start IS NULL OR page_start > 0", name="ck_chunks_page_start"
        ),
        CheckConstraint(
            "page_end IS NULL OR page_end >= page_start", name="ck_chunks_page_end"
        ),
        CheckConstraint("token_count >= 0", name="ck_chunks_token_count"),
        CheckConstraint(
            "overlap_token_count >= 0", name="ck_chunks_overlap_count"
        ),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_document_pages", "document_id", "page_start", "page_end"),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    passage_text = Column(Text, nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    section_title = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=False)
    token_count = Column(Integer, nullable=False)
    overlap_token_count = Column(Integer, nullable=False, default=0, server_default="0")
    chunking_version = Column(String, nullable=False)
    embedding_model = Column(String, nullable=False)
    boundary_kinds = Column(JSONB, nullable=False, default=list, server_default="[]")
    search_vector = Column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(section_title, '')), 'A') "
            "|| setweight(to_tsvector('english', coalesce(passage_text, '')), 'B')",
            persisted=True,
        ),
    )
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document = relationship("Document", back_populates="chunks")

    @property
    def filename(self) -> str:
        """Compatibility view for code that displays a chunk's source filename."""

        return self.document.original_filename
