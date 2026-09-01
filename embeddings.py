from openai import OpenAI
from dotenv import load_dotenv
import os

from settings import APP_SETTINGS

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = APP_SETTINGS.embedding_model


def get_embeddings(
    texts: list[str],
    batch_size: int = 64,
    *,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when supplied")
    if max_retries is not None and max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    client_options = {}
    if timeout_seconds is not None:
        client_options["timeout"] = timeout_seconds
    if max_retries is not None:
        client_options["max_retries"] = max_retries
    request_client = client.with_options(**client_options) if client_options else client
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = request_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start : start + batch_size],
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in ordered)
    if len(embeddings) != len(texts):
        raise RuntimeError("embedding response count did not match input count")
    return embeddings


def get_embedding(
    text: str,
    *,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> list[float]:
    return get_embeddings(
        [text],
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )[0]
