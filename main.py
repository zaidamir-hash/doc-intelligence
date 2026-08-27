from fastapi import FastAPI, UploadFile, File, Depends, Request, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, get_db
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from models import DocumentChunk
from embeddings import get_embedding
from query import search_similar_chunks, generate_answer
from pydantic import BaseModel
from auth import verify_api_key
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pdf_processing import (
    extract_pdf,
    summarize_extraction,
)
from chunking import ChunkingConfig, chunk_extraction, summarize_chunks


PRODUCTION_CHUNK_CONFIG = ChunkingConfig(
    max_tokens=200,
    overlap_tokens=30,
    min_chunk_tokens=40,
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

        extraction = extract_pdf(contents)
        chunks = chunk_extraction(extraction, PRODUCTION_CHUNK_CONFIG)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found. PDF may be scanned or image-based."
            )

        # delete any existing chunks for this filename before inserting new ones
        # (kept in the same uncommitted transaction as the inserts below —
        # see earlier explanation: nothing is permanent until db.commit() at the end)
        db.query(DocumentChunk).filter(
            DocumentChunk.filename == file.filename
        ).delete()

        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk.content)
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Embedding service failed: {str(e)}"
                )

            db_chunk = DocumentChunk(
                filename=file.filename,
                chunk_index=i,
                content=chunk.content,
                embedding=embedding
            )
            db.add(db_chunk)

        db.commit()

        return {
            "filename": file.filename,
            "pages": extraction.page_count,
            "chunks_stored": len(chunks),
            "extraction": summarize_extraction(extraction),
            "chunking": {
                "configuration": {
                    "max_tokens": PRODUCTION_CHUNK_CONFIG.max_tokens,
                    "overlap_tokens": PRODUCTION_CHUNK_CONFIG.overlap_tokens,
                    "min_chunk_tokens": PRODUCTION_CHUNK_CONFIG.min_chunk_tokens,
                    "encoding_name": PRODUCTION_CHUNK_CONFIG.encoding_name,
                    "version": PRODUCTION_CHUNK_CONFIG.version,
                },
                "summary": summarize_chunks(chunks),
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
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
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
