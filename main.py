from fastapi import FastAPI, UploadFile, File, Depends, Request, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, get_db
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from query import search_similar_chunks, generate_answer
from pydantic import BaseModel
from auth import verify_api_key
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pdf_processing import summarize_extraction
from chunking import PRODUCTION_CHUNK_CONFIG, summarize_chunks
from ingestion import (
    EmbeddingGenerationError,
    NoExtractableTextError,
    ingest_document,
)

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/upload")
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        try:
            result = ingest_document(
                db,
                file.filename,
                contents,
                PRODUCTION_CHUNK_CONFIG,
            )
        except NoExtractableTextError as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            )
        except EmbeddingGenerationError as error:
            raise HTTPException(
                status_code=502,
                detail=f"Embedding service failed: {str(error)}",
            )

        return {
            "document_id": str(result.document_id),
            "document_content_hash": result.document_content_hash,
            "filename": result.filename,
            "status": "ready",
            "reused_existing_index": result.reused_existing_index,
            "reindexed_existing_document": result.reindexed_existing_document,
            "pages": result.extraction.page_count,
            "chunks_stored": len(result.chunks),
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing document: {str(e)}"
        )

class QueryRequest(BaseModel):
    question: str
    filename: str

@app.post("/query")
@limiter.limit("20/minute")
def query_document(
    request: Request,
    body: QueryRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    try:
        if not body.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        chunks = search_similar_chunks(body.question, db, body.filename)

        if not chunks:
            return {"answer": "No relevant documents found. Please upload a document first."}

        try:
            answer = generate_answer(body.question, chunks)
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Answer generation failed: {str(e)}"
            )

        sources = [
            {
                "document_id": str(chunk.document_id),
                "document_content_hash": chunk.document.content_hash,
                "chunk_id": str(chunk.id),
                "chunk_content_hash": chunk.content_hash,
                "filename": chunk.document.original_filename,
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_title": chunk.section_title,
                "preview": chunk.content[:100]
            }
            for chunk in chunks
        ]

        return {
            "question": body.question,
            "answer": answer,
            "sources": sources
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing query: {str(e)}"
        )
