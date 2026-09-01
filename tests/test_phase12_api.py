from __future__ import annotations

import asyncio
import io
import os
import unittest
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException, UploadFile

import auth
import main
from database import SessionLocal, init_db
from models import (
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_READY,
    Document,
)
from settings import APP_SETTINGS


def document(status: str = DOCUMENT_STATUS_READY):
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid.uuid4(),
        original_filename="safe.pdf",
        content_hash="a" * 64,
        status=status,
        page_count=3,
        chunk_count=7,
        processing_error="provider error containing a secret",
        created_at=now,
        updated_at=now,
    )


class Phase12ValidationTests(unittest.TestCase):
    def test_filename_validation_uses_safe_pdf_basename(self) -> None:
        self.assertEqual(main._safe_filename(r"C:\unsafe\report.PDF"), "report.PDF")
        with self.assertRaises(HTTPException) as raised:
            main._safe_filename("notes.txt")
        self.assertEqual(raised.exception.status_code, 415)
        self.assertEqual(raised.exception.detail["code"], "unsupported_file_type")

    def test_content_type_and_pdf_signature_are_checked(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            main._validate_pdf_content_type("text/plain")
        self.assertEqual(raised.exception.status_code, 415)

        upload = UploadFile(filename="fake.pdf", file=io.BytesIO(b"not a pdf"))
        with self.assertRaises(HTTPException) as signature_error:
            asyncio.run(main._read_pdf_with_limit(upload))
        self.assertEqual(signature_error.exception.detail["code"], "invalid_pdf_signature")

    def test_failed_document_payload_redacts_internal_error(self) -> None:
        payload = main.document_payload(document("failed"))
        self.assertNotIn("provider error", payload["processing_error"])
        self.assertNotIn("content_hash", payload)

    def test_public_configuration_contains_no_api_key(self) -> None:
        public = APP_SETTINGS.public_configuration()
        if APP_SETTINGS.api_key:
            self.assertNotIn(APP_SETTINGS.api_key, str(public))
        self.assertNotIn("database_url", str(public).lower())
        self.assertNotIn("api_key", public["api"])
        self.assertEqual(
            public["retrieval"]["active_mode"],
            "hybrid_rrf_rerank_expansion",
        )

    def test_authentication_errors_are_structured_and_do_not_echo_key(self) -> None:
        supplied = "definitely-not-the-configured-key"
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(auth.verify_api_key(supplied))
        self.assertIn(raised.exception.status_code, (403, 503))
        self.assertNotIn(supplied, str(raised.exception.detail))


class FakeDatabase:
    def __init__(self, stored_document):
        self.stored_document = stored_document
        self.deleted = None
        self.committed = False

    def get(self, _model, document_id):
        if self.stored_document and self.stored_document.id == document_id:
            return self.stored_document
        return None

    def delete(self, stored_document):
        self.deleted = stored_document

    def commit(self):
        self.committed = True


class Phase12EndpointLogicTests(unittest.TestCase):
    def test_delete_uses_stable_id_and_blocks_processing_document(self) -> None:
        ready = document()
        database = FakeDatabase(ready)
        response = main.delete_document(ready.id, database, "test-key")
        self.assertEqual(response["deleted"]["document_id"], str(ready.id))
        self.assertIs(database.deleted, ready)
        self.assertTrue(database.committed)

        processing = document(DOCUMENT_STATUS_PROCESSING)
        with self.assertRaises(HTTPException) as raised:
            main.delete_document(processing.id, FakeDatabase(processing), "test-key")
        self.assertEqual(raised.exception.status_code, 409)

    def test_query_resolves_stable_id_then_scopes_pipeline_to_exact_version(self) -> None:
        ready = document()
        database = FakeDatabase(ready)
        request_body = main.QueryRequest(
            question="  What happened?  ",
            document_id=ready.id,
            debug=True,
        )
        pipeline_result = object()

        with (
            patch("main.answer_document_question", return_value=pipeline_result) as answer,
            patch("main.build_query_response", return_value={"status": "answered"}) as build,
        ):
            endpoint = getattr(main.query_document, "__wrapped__", main.query_document)
            response = endpoint(Mock(), request_body, database, "test-key")

        self.assertEqual(response, {"status": "answered"})
        answer.assert_called_once_with(
            "What happened?",
            database,
            ready.original_filename,
            document_content_hash=ready.content_hash,
        )
        self.assertEqual(build.call_args.kwargs["document_id"], str(ready.id))
        self.assertTrue(build.call_args.kwargs["include_debug"])


@unittest.skipUnless(
    os.getenv("RUN_DATABASE_TESTS") == "1",
    "set RUN_DATABASE_TESTS=1 to run PostgreSQL integration tests",
)
class Phase12PostgresEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.database = SessionLocal()
        self.filename = f"phase12-api-{uuid.uuid4()}.pdf"
        self.stored = Document(
            original_filename=self.filename,
            content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
            status=DOCUMENT_STATUS_READY,
            page_count=2,
            chunk_count=0,
            extraction_version="phase12-test",
            chunking_version="phase12-test",
            chunking_config={},
            embedding_model=APP_SETTINGS.embedding_model,
        )
        self.database.add(self.stored)
        self.database.commit()
        self.database.refresh(self.stored)

    def tearDown(self) -> None:
        self.database.rollback()
        self.database.query(Document).filter(
            Document.original_filename == self.filename
        ).delete(synchronize_session=False)
        self.database.commit()
        self.database.close()

    def test_document_persists_in_list_and_delete_removes_exact_id(self) -> None:
        listing = main.list_documents(self.database, "test-key")
        listed_ids = {item["document_id"] for item in listing["documents"]}
        self.assertIn(str(self.stored.id), listed_ids)

        main.delete_document(self.stored.id, self.database, "test-key")
        self.assertIsNone(self.database.get(Document, self.stored.id))


if __name__ == "__main__":
    unittest.main()
