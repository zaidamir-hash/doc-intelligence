from __future__ import annotations

import os
import unittest
import uuid
from unittest.mock import patch

from chunking import ChunkingConfig
from database import SessionLocal, init_db
from ingestion import EmbeddingGenerationError, ingest_document
from models import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_READY,
    DOCUMENT_STATUS_SUPERSEDED,
    Document,
    DocumentChunk,
)
from pdf_processing import process_extracted_pages


@unittest.skipUnless(
    os.getenv("RUN_DATABASE_TESTS") == "1",
    "set RUN_DATABASE_TESTS=1 to run PostgreSQL integration tests",
)
class IngestionPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.database = SessionLocal()
        self.filename = f"phase4-test-{uuid.uuid4()}.pdf"
        self.config = ChunkingConfig(
            max_tokens=100,
            overlap_tokens=10,
            min_chunk_tokens=20,
        )
        self.extraction = process_extracted_pages(
            ["1 Test Section\n\nStable evidence for the integration test."]
        )

    def tearDown(self) -> None:
        self.database.rollback()
        self.database.query(Document).filter(
            Document.original_filename == self.filename
        ).delete(synchronize_session=False)
        self.database.commit()
        self.database.close()

    @staticmethod
    def embedding(_: str) -> list[float]:
        return [0.0] * 1536

    def ingest(self, contents: bytes, *, force: bool = False):
        with patch("ingestion.extract_pdf", return_value=self.extraction):
            return ingest_document(
                self.database,
                self.filename,
                contents,
                self.config,
                embedding_function=self.embedding,
                force=force,
            )

    def test_idempotency_supersession_and_metadata_persistence(self) -> None:
        first = self.ingest(b"first PDF bytes")
        stored = self.database.get(Document, first.document_id)
        chunk = (
            self.database.query(DocumentChunk)
            .filter(DocumentChunk.document_id == first.document_id)
            .one()
        )

        self.assertEqual(stored.status, DOCUMENT_STATUS_READY)
        self.assertEqual(stored.chunk_count, 1)
        self.assertEqual(stored.chunking_config["max_tokens"], 100)
        self.assertEqual(stored.chunking_config["overlap_tokens"], 10)
        self.assertEqual(chunk.page_start, 1)
        self.assertEqual(chunk.page_end, 1)
        self.assertEqual(chunk.section_title, "1 Test Section")
        self.assertGreater(chunk.token_count, 0)
        self.assertEqual(len(chunk.content_hash), 64)
        self.assertIsNotNone(chunk.search_vector)

        def unexpected_embedding(_: str) -> list[float]:
            raise AssertionError("current idempotent upload must not call embeddings")

        with patch("ingestion.extract_pdf", return_value=self.extraction):
            reused = ingest_document(
                self.database,
                self.filename,
                b"first PDF bytes",
                self.config,
                embedding_function=unexpected_embedding,
            )
        self.assertTrue(reused.reused_existing_index)
        self.assertEqual(reused.document_id, first.document_id)

        replacement = self.ingest(b"different PDF bytes")
        self.database.refresh(stored)
        self.assertEqual(stored.status, DOCUMENT_STATUS_SUPERSEDED)
        self.assertEqual(
            self.database.get(Document, replacement.document_id).status,
            DOCUMENT_STATUS_READY,
        )

    def test_failure_exposes_no_partial_chunks_and_preserves_ready_index(self) -> None:
        ready = self.ingest(b"ready PDF bytes")
        original_hashes = [
            row.content_hash
            for row in self.database.query(DocumentChunk)
            .filter(DocumentChunk.document_id == ready.document_id)
            .order_by(DocumentChunk.chunk_index)
        ]

        def fail_embedding(_: str) -> list[float]:
            raise RuntimeError("simulated embedding outage")

        with patch("ingestion.extract_pdf", return_value=self.extraction):
            with self.assertRaises(EmbeddingGenerationError):
                ingest_document(
                    self.database,
                    self.filename,
                    b"failed replacement bytes",
                    self.config,
                    embedding_function=fail_embedding,
                )

        failed = (
            self.database.query(Document)
            .filter(
                Document.original_filename == self.filename,
                Document.status == DOCUMENT_STATUS_FAILED,
            )
            .one()
        )
        self.assertEqual(failed.chunk_count, 0)
        self.assertEqual(len(failed.chunks), 0)
        self.assertEqual(self.database.get(Document, ready.document_id).status, "ready")

        with patch("ingestion.extract_pdf", return_value=self.extraction):
            with self.assertRaises(EmbeddingGenerationError):
                ingest_document(
                    self.database,
                    self.filename,
                    b"ready PDF bytes",
                    self.config,
                    embedding_function=fail_embedding,
                    force=True,
                )

        current_hashes = [
            row.content_hash
            for row in self.database.query(DocumentChunk)
            .filter(DocumentChunk.document_id == ready.document_id)
            .order_by(DocumentChunk.chunk_index)
        ]
        self.assertEqual(current_hashes, original_hashes)
        self.assertEqual(self.database.get(Document, ready.document_id).status, "ready")

    def test_incomplete_embedding_batch_is_recorded_as_failed(self) -> None:
        with (
            patch("ingestion.extract_pdf", return_value=self.extraction),
            patch("ingestion.get_embeddings", return_value=[]),
        ):
            with self.assertRaisesRegex(
                EmbeddingGenerationError,
                "different number of vectors",
            ):
                ingest_document(
                    self.database,
                    self.filename,
                    b"incomplete embedding batch",
                    self.config,
                )

        failed = (
            self.database.query(Document)
            .filter(
                Document.original_filename == self.filename,
                Document.status == DOCUMENT_STATUS_FAILED,
            )
            .one()
        )
        self.assertEqual(failed.chunk_count, 0)
        self.assertEqual(failed.chunks, [])


if __name__ == "__main__":
    unittest.main()
