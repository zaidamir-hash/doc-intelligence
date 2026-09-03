"""Controlled, inspectable query expansion that never replaces the original."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI
from pydantic import BaseModel, Field

from reranking import estimate_cost_usd
from settings import APP_SETTINGS


DEFAULT_EXPANSION_MODEL = APP_SETTINGS.expansion_model
DEFAULT_REQUEST_TIMEOUT_SECONDS = APP_SETTINGS.model_request_timeout_seconds
DEFAULT_MAX_RETRIES = APP_SETTINGS.model_max_retries
MAX_EXPANSION_CHARACTERS = 500

_QUOTED_PHRASE = re.compile(r'"([^"\r\n]+)"')
_TOKEN = re.compile(
    r"[A-Za-z0-9]+(?:[+./,'’-][A-Za-z0-9]+|\([A-Za-z0-9]+\))*\+?",
    re.UNICODE,
)
_CAPITALIZED_SEQUENCE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:['’+.-][A-Za-z0-9]+)*)(?:\s+"
    r"(?:[A-Z][A-Za-z0-9]*(?:['’+.-][A-Za-z0-9]+)*)){0,5}\b"
)
_QUESTION_PREFIXES = {
    "after",
    "by",
    "how",
    "in",
    "to",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class ExpansionUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(frozen=True)
class GeneratedExpansion:
    alternative_query: str
    lexical_terms: tuple[str, ...]
    intent_preserved: bool
    rationale: str
    usage: ExpansionUsage = ExpansionUsage()


@dataclass(frozen=True)
class QueryExpansionResult:
    original_query: str
    generated_query: str | None
    expanded_query: str | None
    lexical_terms: tuple[str, ...]
    protected_terms: tuple[str, ...]
    used_expansion: bool
    fallback_reason: str | None
    rationale: str | None
    usage: ExpansionUsage
    latency_ms: float


class _StructuredExpansion(BaseModel):
    alternative_query: str = Field(min_length=1, max_length=MAX_EXPANSION_CHARACTERS)
    lexical_terms: list[str] = Field(max_length=12)
    intent_preserved: bool
    rationale: str


GenerateExpansionFunction = Callable[[str, tuple[str, ...], str], GeneratedExpansion]


def _deduplicate(values: list[str]) -> tuple[str, ...]:
    unique: dict[str, str] = {}
    for value in values:
        cleaned = " ".join(value.strip().split())
        if cleaned:
            unique.setdefault(cleaned.casefold(), cleaned)
    return tuple(unique.values())


def _is_strong_exact_token(token: str) -> bool:
    has_digit = any(character.isdigit() for character in token)
    has_identifier_punctuation = any(character in token for character in "+/(),.-")
    is_acronym = len(token) >= 2 and token.isupper() and any(
        character.isalpha() for character in token
    )
    return has_digit or has_identifier_punctuation or is_acronym


def extract_protected_terms(question: str) -> tuple[str, ...]:
    """Extract exact signals whose removal could change retrieval intent."""

    if not question.strip():
        raise ValueError("query cannot be empty")
    phrases = list(_QUOTED_PHRASE.findall(question))
    tokens = _TOKEN.findall(_QUOTED_PHRASE.sub(" ", question))
    strong_tokens = [token for token in tokens if _is_strong_exact_token(token)]
    names = []
    for match in _CAPITALIZED_SEQUENCE.findall(question):
        words = match.split()
        while words and words[0].casefold() in _QUESTION_PREFIXES:
            words.pop(0)
        if words:
            names.append(" ".join(words))
    return _deduplicate(phrases + strong_tokens + names)


def _content_terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN.findall(text)
        if len(token) >= 3 and token.casefold() not in _STOP_WORDS
    }


def validate_expansion(
    original_query: str,
    generated: GeneratedExpansion,
    protected_terms: tuple[str, ...] | None = None,
) -> str:
    """Return a cleaned expansion or raise with an inspectable safety reason."""

    original = " ".join(original_query.strip().split())
    expansion = " ".join(generated.alternative_query.strip().split())
    if not expansion:
        raise ValueError("generated expansion was empty")
    if len(expansion) > MAX_EXPANSION_CHARACTERS:
        raise ValueError("generated expansion exceeded the character limit")
    if expansion.casefold() == original.casefold():
        raise ValueError("generated expansion was identical to the original")
    if not generated.intent_preserved:
        raise ValueError("generator reported that intent was not preserved")

    protected = protected_terms or extract_protected_terms(original)
    missing = [
        term for term in protected if term.casefold() not in expansion.casefold()
    ]
    if missing:
        raise ValueError(f"generated expansion removed protected terms: {missing}")

    original_strong = {
        token.casefold()
        for token in _TOKEN.findall(original)
        if _is_strong_exact_token(token)
    }
    expansion_strong = {
        token.casefold()
        for token in _TOKEN.findall(expansion)
        if _is_strong_exact_token(token)
    }
    novel_strong = sorted(expansion_strong - original_strong)
    if novel_strong:
        raise ValueError(
            f"generated expansion introduced new exact signals: {novel_strong}"
        )
    if not (_content_terms(original) & _content_terms(expansion)):
        raise ValueError("generated expansion had no content-word overlap")
    return expansion


def generate_expansion_openai(
    question: str,
    protected_terms: tuple[str, ...],
    model: str = DEFAULT_EXPANSION_MODEL,
    *,
    client: OpenAI | None = None,
) -> GeneratedExpansion:
    """Generate one retrieval-oriented alternative with structured output."""

    api_client = client or OpenAI(
        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )
    response = api_client.responses.parse(
        model=model,
        instructions=(
            "Create exactly one concise retrieval-oriented alternative query. "
            "Preserve the user's intent and every protected term verbatim. Use "
            "synonyms or likely document terminology to reduce vocabulary mismatch. "
            "Do not answer the question. Do not add facts, names, numbers, dates, "
            "acronyms, or identifiers. Return a short list of important lexical "
            "terms already present in or safely paraphrased from the question."
        ),
        input=(
            f"Original query:\n{question}\n\n"
            f"Protected terms:\n{list(protected_terms)}"
        ),
        text_format=_StructuredExpansion,
        temperature=0,
        store=False,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("expansion model returned no parsed structured output")
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    return GeneratedExpansion(
        alternative_query=parsed.alternative_query,
        lexical_terms=_deduplicate(parsed.lexical_terms),
        intent_preserved=parsed.intent_preserved,
        rationale=parsed.rationale.strip(),
        usage=ExpansionUsage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost_usd(input_tokens, output_tokens),
        ),
    )


def expand_query(
    question: str,
    *,
    enabled: bool = True,
    model: str = DEFAULT_EXPANSION_MODEL,
    generator: GenerateExpansionFunction = generate_expansion_openai,
) -> QueryExpansionResult:
    """Return one validated expansion or an explicit original-only fallback."""

    original = " ".join(question.strip().split())
    if not original:
        raise ValueError("query cannot be empty")
    protected_terms = extract_protected_terms(original)
    if not enabled:
        return QueryExpansionResult(
            original_query=original,
            generated_query=None,
            expanded_query=None,
            lexical_terms=(),
            protected_terms=protected_terms,
            used_expansion=False,
            fallback_reason="expansion disabled",
            rationale=None,
            usage=ExpansionUsage(),
            latency_ms=0.0,
        )

    started = time.perf_counter()
    generated: GeneratedExpansion | None = None
    try:
        generated = generator(original, protected_terms, model)
        expanded = validate_expansion(original, generated, protected_terms)
        return QueryExpansionResult(
            original_query=original,
            generated_query=generated.alternative_query,
            expanded_query=expanded,
            lexical_terms=generated.lexical_terms,
            protected_terms=protected_terms,
            used_expansion=True,
            fallback_reason=None,
            rationale=generated.rationale,
            usage=generated.usage,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as error:
        return QueryExpansionResult(
            original_query=original,
            generated_query=(
                generated.alternative_query if generated is not None else None
            ),
            expanded_query=None,
            lexical_terms=(generated.lexical_terms if generated is not None else ()),
            protected_terms=protected_terms,
            used_expansion=False,
            fallback_reason=f"{type(error).__name__}: {error}",
            rationale=(generated.rationale if generated is not None else None),
            usage=(generated.usage if generated is not None else ExpansionUsage()),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
