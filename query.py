from sqlalchemy.orm import Session
from models import DocumentChunk
from embeddings import get_embedding
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def search_similar_chunks(query: str, db: Session, top_k: int = 5):
    query_embedding = get_embedding(query)
    
    results = db.query(DocumentChunk).order_by(
        DocumentChunk.embedding.l2_distance(query_embedding)
    ).limit(top_k).all()
    
    return results

def generate_answer(query: str, chunks: list[DocumentChunk]) -> str:
    context = "\n\n".join([chunk.content for chunk in chunks])
    
    prompt = f"""You are a helpful assistant. Answer the question based only on the context provided below.
If the answer cannot be found in the context, say "I cannot find this information in the provided documents."

Context:
{context}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content