from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start : start + batch_size],
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in ordered)
    if len(embeddings) != len(texts):
        raise RuntimeError("embedding response count did not match input count")
    return embeddings


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]
