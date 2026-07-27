from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.security import APIKeyHeader
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
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
def root():
    return {"status": "running"}

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks

@app.post("/upload")
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    contents = await file.read()
    pdf = PdfReader(io.BytesIO(contents))
    
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        
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

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
@limiter.limit("20/minute")
def query_document(
    request: Request,
    body: QueryRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    chunks = search_similar_chunks(body.question, db)
    
    if not chunks:
        return {"answer": "No relevant documents found. Please upload a document first."}
    
    answer = generate_answer(body.question, chunks)
    
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