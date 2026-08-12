from fastapi import FastAPI, UploadFile, File, Depends, Request, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
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
import io

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

def find_break_point(text: str, start: int, end: int) -> int:
    search_window = 200
    search_start = max(start, end - search_window)
    chunk = text[search_start:end]

    # Priority 1: paragraph break
    para_break = chunk.rfind("\n\n")
    if para_break != -1:
        return search_start + para_break + 2

    # Priority 2: sentence end
    for punct in [". ", "! ", "? "]:
        sentence_break = chunk.rfind(punct)
        if sentence_break != -1:
            return search_start + sentence_break + len(punct)

    # Priority 3: single line break (catches bullet points, resumes, headers)
    line_break = chunk.rfind("\n")
    if line_break != -1:
        return search_start + line_break + 1

    # Priority 4: no good boundary found, hard cut
    return end

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            end = find_break_point(text, start, end)

        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())

        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks

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

        pdf = PdfReader(io.BytesIO(contents))

        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

        if not text.strip():
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

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk)
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Embedding service failed: {str(e)}"
                )

            db_chunk = DocumentChunk(
                filename=file.filename,
                chunk_index=i,
                content=chunk,
                embedding=embedding
            )
            db.add(db_chunk)

        db.commit()

        return {
            "filename": file.filename,
            "pages": len(pdf.pages),
            "chunks_stored": len(chunks)
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