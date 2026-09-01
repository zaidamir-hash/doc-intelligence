"""FastAPI integration layer for the Lexis document and query pipelines."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import PurePath
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from auth import verify_api_key
from chunking import PRODUCTION_CHUNK_CONFIG, summarize_chunks
from database import get_db, init_db
from ingestion import EmbeddingGenerationError, NoExtractableTextError, ingest_document
from models import (
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_READY,
    DOCUMENT_STATUS_SUPERSEDED,
    Document,
)
from pdf_processing import summarize_extraction
from query import answer_document_question, build_query_response
from settings import APP_SETTINGS


limiter = Limiter(key_func=get_remote_address)


def _error(status_code: int, code: str, message: str) -> HTTPException:
    """Build a consistent client-safe API error."""

    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _safe_filename(filename: str | None) -> str:
    """Return a basename safe for display and persistent document identity."""

    if not filename or not filename.strip():
        raise _error(400, "missing_filename", "The uploaded file needs a name.")
    if "\x00" in filename:
        raise _error(400, "invalid_filename", "The filename is invalid.")

    cleaned = PurePath(filename.replace("\\", "/")).name.strip()
    if not cleaned or len(cleaned) > APP_SETTINGS.max_filename_characters:
        raise _error(
            400,
            "invalid_filename",
            "The filename is empty or longer than the configured limit.",
        )
    if not cleaned.lower().endswith(".pdf"):
        raise _error(415, "unsupported_file_type", "Only PDF files are supported.")
    return cleaned


def _validate_pdf_content_type(content_type: str | None) -> None:
    if content_type not in APP_SETTINGS.accepted_pdf_content_types:
        allowed = ", ".join(APP_SETTINGS.accepted_pdf_content_types)
        raise _error(
            415,
            "unsupported_content_type",
            f"Upload a PDF using one of these content types: {allowed}.",
        )


async def _read_pdf_with_limit(file: UploadFile) -> bytes:
    """Read at most one byte beyond the limit so oversized uploads are rejected."""

    contents = await file.read(APP_SETTINGS.max_upload_bytes + 1)
    if len(contents) > APP_SETTINGS.max_upload_bytes:
        max_megabytes = APP_SETTINGS.max_upload_bytes / (1024 * 1024)
        raise _error(
            413,
            "file_too_large",
            f"The PDF exceeds the {max_megabytes:g} MB upload limit.",
        )
    if not contents:
        raise _error(400, "empty_file", "The uploaded PDF is empty.")
    if b"%PDF-" not in contents[:1024]:
        raise _error(
            415,
            "invalid_pdf_signature",
            "The uploaded file does not appear to be a valid PDF.",
        )
    return contents


def _safe_processing_error(document: Document) -> str | None:
    """Never expose stored provider errors, paths, or secrets to clients."""

    if document.status != "failed":
        return None
    return "Indexing failed. Re-upload the PDF or check the backend logs."


def document_payload(document: Document) -> dict[str, object]:
    """Serialize lifecycle state without leaking internal provider errors."""

    return {
        "document_id": str(document.id),
        "filename": document.original_filename,
        "status": document.status,
        "pages": document.page_count,
        "chunks_stored": document.chunk_count,
        "processing_error": _safe_processing_error(document),
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "updated_at": document.updated_at.isoformat() if document.updated_at else None,
    }


def _get_document(database: Session, document_id: UUID) -> Document:
    document = database.get(Document, document_id)
    if document is None or document.status == DOCUMENT_STATUS_SUPERSEDED:
        raise _error(404, "document_not_found", "That document was not found.")
    return document


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Lexis API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(APP_SETTINGS.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", APP_SETTINGS.api_key_header],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "running"}


@app.get("/auth/validate")
def validate_authentication(
    api_key: str = Depends(verify_api_key),
) -> dict[str, bool]:
    """Give the frontend a real backend authentication check."""

    return {"authenticated": True}


@app.get("/configuration")
def get_configuration(
    api_key: str = Depends(verify_api_key),
) -> dict[str, object]:
    return APP_SETTINGS.public_configuration()


@app.get("/documents")
def list_documents(
    database: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> dict[str, list[dict[str, object]]]:
    """Return backend-authoritative current document state after refresh."""

    documents = (
        database.query(Document)
        .filter(Document.status != DOCUMENT_STATUS_SUPERSEDED)
        .order_by(Document.updated_at.desc(), Document.created_at.desc())
        .all()
    )
    return {"documents": [document_payload(document) for document in documents]}


@app.get("/documents/{document_id}")
def get_document(
    document_id: UUID,
    database: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> dict[str, object]:
    return document_payload(_get_document(database, document_id))


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: UUID,
    database: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> dict[str, object]:
    """Delete one stable document identity and its chunks via DB cascade."""

    document = _get_document(database, document_id)
    if document.status == DOCUMENT_STATUS_PROCESSING:
        raise _error(
            409,
            "document_processing",
            "Wait for indexing to finish before deleting this document.",
        )
    deleted = document_payload(document)
    database.delete(document)
    database.commit()
    return {"deleted": deleted}


@app.post("/upload")
@limiter.limit(APP_SETTINGS.upload_rate_limit)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    database: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> dict[str, object]:
    filename = _safe_filename(file.filename)
    _validate_pdf_content_type(file.content_type)
    contents = await _read_pdf_with_limit(file)

    try:
        result = ingest_document(
            database,
            filename,
            contents,
            PRODUCTION_CHUNK_CONFIG,
        )
    except NoExtractableTextError as error:
        raise _error(
            422,
            "no_extractable_text",
            "No extractable text was found. This PDF may be scanned or "
            "image-only; OCR is not enabled.",
        ) from error
    except EmbeddingGenerationError as error:
        raise _error(
            502,
            "embedding_service_failed",
            "The embedding service could not index this PDF. Try again later.",
        ) from error
    except Exception as error:
        database.rollback()
        raise _error(
            500,
            "document_processing_failed",
            "The PDF could not be processed. Check the backend logs.",
        ) from error

    document = _get_document(database, result.document_id)
    return {
        **document_payload(document),
        "document_content_hash": result.document_content_hash,
        "reused_existing_index": result.reused_existing_index,
        "reindexed_existing_document": result.reindexed_existing_document,
        "extraction": summarize_extraction(result.extraction),
        "chunking": {
            "configuration": {
                "max_tokens": PRODUCTION_CHUNK_CONFIG.max_tokens,
                "overlap_tokens": PRODUCTION_CHUNK_CONFIG.overlap_tokens,
                "min_chunk_tokens": PRODUCTION_CHUNK_CONFIG.min_chunk_tokens,
                "encoding_name": PRODUCTION_CHUNK_CONFIG.encoding_name,
                "version": PRODUCTION_CHUNK_CONFIG.version,
            },
            "summary": summarize_chunks(list(result.chunks)),
        },
    }


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: UUID
    debug: bool = False


@app.post("/query")
@limiter.limit(APP_SETTINGS.query_rate_limit)
def query_document(
    request: Request,
    body: QueryRequest,
    database: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> dict[str, object]:
    question = body.question.strip()
    if not question:
        raise _error(400, "empty_question", "Question cannot be empty.")

    document = _get_document(database, body.document_id)
    if document.status != DOCUMENT_STATUS_READY:
        raise _error(
            409,
            "document_not_ready",
            "The selected document is not ready to query.",
        )

    try:
        result = answer_document_question(
            question,
            database,
            document.original_filename,
            document_content_hash=document.content_hash,
        )
    except Exception as error:
        raise _error(
            502,
            "query_pipeline_failed",
            "The retrieval or answer service failed. Try again later.",
        ) from error

    return build_query_response(
        question,
        result,
        document_id=str(document.id),
        retrieval_mode=APP_SETTINGS.retrieval_mode,
        include_debug=body.debug,
        retrieval_configuration=APP_SETTINGS.public_configuration()["retrieval"],
    )
