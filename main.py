from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from database import init_db
from contextlib import asynccontextmanager
import io
from embeddings import get_embedding
from database import get_db
from sqlalchemy.orm import Session
from models import DocumentChunk
from fastapi import Depends

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "running"}

# chunk function BEFORE the upload endpoint
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

# upload endpoint AFTER the chunk function
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
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