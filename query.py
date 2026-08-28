from sqlalchemy.orm import Session
from dense_retrieval import DenseCandidate, retrieve_dense_candidates, select_dense_context
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def search_similar_chunks(
    query: str,
    db: Session,
    filename: str,
    top_k: int = 5,
    candidate_k: int = 30,
) -> list[DenseCandidate]:
    """Retrieve broad dense candidates, then choose non-duplicate context."""

    candidates = retrieve_dense_candidates(
        query,
        db,
        filename,
        candidate_k=candidate_k,
        metric="cosine",
    )
    return list(
        select_dense_context(
            candidates,
            top_k=top_k,
            duplicate_similarity_threshold=0.9,
        ).selected
    )

def generate_answer(query: str, chunks: list[DenseCandidate]) -> str:
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
