from __future__ import annotations

import os
import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from database import SessionLocal, get_db, init_db
from dense_retrieval import retrieve_dense_candidates
from lexical_retrieval import retrieve_lexical_candidates
from models import Document, DocumentChunk
from pdf_processing import process_extracted_pages


@unittest.skipUnless(
    os.getenv("RUN_DATABASE_TESTS") == "1",
    "set RUN_DATABASE_TESTS=1 to run PostgreSQL integration tests",
)
class EndToEndPostgresTests(unittest.TestCase):
    """Exercise the HTTP lifecycle plus real PostgreSQL dense/lexical retrieval."""

    @classmethod
    def setUpClass(cls) -> None:
        init_db()

        def database_override():
            database = SessionLocal()
            try:
                yield database
            finally:
                database.close()

        main.app.dependency_overrides[get_db] = database_override
        main.app.dependency_overrides[main.verify_api_key] = lambda: "test-key"
        cls.client_context = TestClient(main.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        main.app.dependency_overrides.clear()

    def setUp(self) -> None:
        self.filename = f"phase13-e2e-{uuid.uuid4()}.pdf"

    def tearDown(self) -> None:
        database = SessionLocal()
        try:
            database.query(Document).filter(
                Document.original_filename == self.filename
            ).delete(synchronize_session=False)
            database.commit()
        finally:
            database.close()

    def test_upload_list_retrieve_query_and_delete(self) -> None:
        extraction = process_extracted_pages(
            [
                "1 Integration Evidence\n\n"
                "The ALPHA-42 programme records a verified contract value of "
                "75 million rupees. This fixed paragraph exists only to exercise "
                "the final HTTP ingestion lifecycle, stable document identity, "
                "page provenance, PostgreSQL lexical search, pgvector dense "
                "search, query routing, and cascading deletion without calling "
                "an external model during the integration test."
            ]
        )

        with (
            patch("ingestion.extract_pdf", return_value=extraction),
            patch(
                "ingestion.get_embeddings",
                side_effect=lambda texts: [
                    [1.0] + [0.0] * 1535 for _ in texts
                ],
            ),
        ):
            upload = self.client.post(
                "/upload",
                files={"file": (self.filename, b"%PDF-1.7\nphase13", "application/pdf")},
            )

        self.assertEqual(upload.status_code, 200, upload.text)
        uploaded = upload.json()
        document_id = uploaded["document_id"]
        document_hash = uploaded["document_content_hash"]
        self.assertEqual(uploaded["status"], "ready")
        self.assertGreater(uploaded["chunks_stored"], 0)

        listing = self.client.get("/documents")
        self.assertEqual(listing.status_code, 200)
        matching = [
            item
            for item in listing.json()["documents"]
            if item["document_id"] == document_id
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["status"], "ready")

        database = SessionLocal()
        try:
            dense = retrieve_dense_candidates(
                "What is the ALPHA-42 contract value?",
                database,
                self.filename,
                candidate_k=5,
                document_content_hash=document_hash,
                embedding_function=lambda _query: [1.0] + [0.0] * 1535,
            )
            lexical = retrieve_lexical_candidates(
                "What is the ALPHA-42 contract value?",
                database,
                self.filename,
                candidate_k=5,
                document_content_hash=document_hash,
            )
        finally:
            database.close()
        self.assertTrue(dense)
        self.assertTrue(lexical)
        self.assertEqual(dense[0].document_id, document_id)
        self.assertEqual(lexical[0].document_id, document_id)

        with (
            patch("main.answer_document_question", return_value=object()) as answer,
            patch(
                "main.build_query_response",
                return_value={
                    "status": "answered",
                    "answer": "The value is 75 million rupees.",
                    "citations": [{"page_start": 1, "page_end": 1}],
                    "evidence": [{"page_start": 1, "page_end": 1}],
                },
            ),
        ):
            query = self.client.post(
                "/query",
                json={
                    "document_id": document_id,
                    "question": "What is the ALPHA-42 contract value?",
                    "debug": False,
                },
            )
        self.assertEqual(query.status_code, 200, query.text)
        self.assertEqual(query.json()["citations"][0]["page_start"], 1)
        self.assertEqual(
            answer.call_args.kwargs["document_content_hash"],
            document_hash,
        )

        deletion = self.client.delete(f"/documents/{document_id}")
        self.assertEqual(deletion.status_code, 200, deletion.text)
        database = SessionLocal()
        try:
            self.assertIsNone(database.get(Document, uuid.UUID(document_id)))
            remaining_chunks = (
                database.query(DocumentChunk)
                .filter(DocumentChunk.document_id == uuid.UUID(document_id))
                .count()
            )
        finally:
            database.close()
        self.assertEqual(remaining_chunks, 0)


if __name__ == "__main__":
    unittest.main()
