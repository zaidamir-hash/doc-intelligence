"""Independent PostgreSQL full-text candidate retrieval for Lexis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Integer, case, func, literal
from sqlalchemy.orm import Session

from models import DOCUMENT_STATUS_READY, Document, DocumentChunk


TEXT_SEARCH_CONFIGURATION = "english"
RANK_NORMALIZATION = 32
DEFAULT_EXACT_MATCH_BOOST = 0.25

_QUOTED_PHRASE = re.compile(r'"([^"\r\n]+)"')
_QUERY_TOKEN = re.compile(
    r"[A-Za-z0-9]+(?:[+./,'’-][A-Za-z0-9]+|\([A-Za-z0-9]+\))*\+?",
    re.UNICODE,
)


@dataclass(frozen=True)
class ParsedLexicalQuery:
    """Safe web-search input plus terms eligible for literal supplementation."""

    search_text: str
    exact_terms: tuple[str, ...]


@dataclass(frozen=True)
class LexicalCandidate:
    """One lexical result with stable provenance and transparent scores."""

    chunk_id: str
    chunk_content_hash: str
    document_id: str
    document_content_hash: str
    chunk_index: int
    filename: str
    content: str
    passage_text: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    lexical_rank: int
    lexical_score: float
    fts_score: float
    exact_match_count: int
    exact_terms_matched: tuple[str, ...]
    parsed_search_text: str


def _deduplicate(values: list[str]) -> tuple[str, ...]:
    unique: dict[str, str] = {}
    for value in values:
        cleaned = value.strip()
        if cleaned:
            unique.setdefault(cleaned.casefold(), cleaned)
    return tuple(unique.values())


def _is_exact_signal(token: str) -> bool:
    """Keep acronyms and punctuation-bearing identifiers, not ordinary words."""

    has_identifier_punctuation = any(character in token for character in "+/(),.-")
    is_acronym = len(token) >= 3 and token.isupper() and any(
        character.isalpha() for character in token
    )
    return has_identifier_punctuation or is_acronym


def parse_lexical_query(question: str) -> ParsedLexicalQuery:
    """Turn a natural-language question into safe OR-style web-search syntax.

    PostgreSQL's `websearch_to_tsquery` normally joins plain words with AND.
    Questions often contain explanatory words that do not appear in the same
    passage, so Lexis joins independently useful clauses with explicit OR.
    Quoted user phrases remain quoted phrases.
    """

    if not question.strip():
        raise ValueError("query cannot be empty")

    phrases = _deduplicate(_QUOTED_PHRASE.findall(question))
    without_phrases = _QUOTED_PHRASE.sub(" ", question)
    tokens = _deduplicate(_QUERY_TOKEN.findall(without_phrases))
    clauses = [f'"{phrase}"' for phrase in phrases]
    clauses.extend(tokens)
    if not clauses:
        raise ValueError("query contains no searchable text")

    exact_terms = _deduplicate(
        list(phrases) + [token for token in tokens if _is_exact_signal(token)]
    )
    return ParsedLexicalQuery(
        search_text=" OR ".join(clauses),
        exact_terms=exact_terms,
    )


def retrieve_lexical_candidates(
    query: str,
    db: Session,
    filename: str,
    *,
    candidate_k: int = 30,
    document_content_hash: str | None = None,
    exact_matching: Literal["disabled", "supplement"] = "supplement",
    exact_match_boost: float = DEFAULT_EXACT_MATCH_BOOST,
) -> list[LexicalCandidate]:
    """Retrieve chunks using only PostgreSQL lexical signals—never embeddings."""

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if exact_matching not in ("disabled", "supplement"):
        raise ValueError("exact_matching must be 'disabled' or 'supplement'")
    if exact_match_boost < 0:
        raise ValueError("exact_match_boost cannot be negative")

    parsed = parse_lexical_query(query)
    ts_query = func.websearch_to_tsquery(
        TEXT_SEARCH_CONFIGURATION, parsed.search_text
    )
    fts_score = func.ts_rank_cd(
        DocumentChunk.search_vector, ts_query, RANK_NORMALIZATION
    )
    fts_matches = DocumentChunk.search_vector.bool_op("@@")(ts_query)

    exact_expressions = [
        case(
            (
                func.strpos(
                    func.lower(DocumentChunk.passage_text), literal(term.casefold())
                )
                > 0,
                1,
            ),
            else_=0,
        ).cast(Integer)
        for term in parsed.exact_terms
    ]
    exact_match_count = sum(exact_expressions, literal(0))
    use_exact_matching = exact_matching == "supplement" and bool(exact_expressions)
    document_filters = [
        Document.original_filename == filename,
        Document.status == DOCUMENT_STATUS_READY,
    ]
    if document_content_hash:
        document_filters.append(Document.content_hash == document_content_hash)

    fts_rows = (
        db.query(
            DocumentChunk,
            Document,
            fts_score.label("fts_score"),
        )
        .join(Document)
        .filter(*document_filters, fts_matches)
        .order_by(fts_score.desc(), DocumentChunk.chunk_index)
        .limit(candidate_k)
        .all()
    )

    rows_by_chunk_id = {
        str(chunk.id): (chunk, document, float(raw_fts))
        for chunk, document, raw_fts in fts_rows
    }
    if use_exact_matching:
        exact_rows = (
            db.query(
                DocumentChunk,
                Document,
                fts_score.label("fts_score"),
                exact_match_count.label("exact_match_count"),
            )
            .join(Document)
            .filter(*document_filters, exact_match_count > 0)
            .order_by(
                exact_match_count.desc(),
                fts_score.desc(),
                DocumentChunk.chunk_index,
            )
            .limit(candidate_k)
            .all()
        )
        for chunk, document, raw_fts, _raw_exact_count in exact_rows:
            rows_by_chunk_id[str(chunk.id)] = (chunk, document, float(raw_fts))

    scored_rows = []
    for chunk, document, raw_fts in rows_by_chunk_id.values():
        matched_terms = tuple(
            term
            for term in parsed.exact_terms
            if term.casefold() in chunk.passage_text.casefold()
        )
        exact_count = len(matched_terms) if use_exact_matching else 0
        score = raw_fts + exact_count * exact_match_boost
        scored_rows.append(
            (score, raw_fts, exact_count, chunk, document, matched_terms)
        )
    scored_rows.sort(
        key=lambda row: (-row[0], -row[1], row[3].chunk_index)
    )

    candidates = []
    for rank, (
        raw_score,
        raw_fts,
        raw_exact_count,
        chunk,
        document,
        matched_terms,
    ) in enumerate(
        scored_rows[:candidate_k], start=1
    ):
        candidates.append(
            LexicalCandidate(
                chunk_id=str(chunk.id),
                chunk_content_hash=chunk.content_hash,
                document_id=str(document.id),
                document_content_hash=document.content_hash,
                chunk_index=chunk.chunk_index,
                filename=document.original_filename,
                content=chunk.content,
                passage_text=chunk.passage_text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
                lexical_rank=rank,
                lexical_score=float(raw_score),
                fts_score=float(raw_fts),
                exact_match_count=int(raw_exact_count),
                exact_terms_matched=matched_terms,
                parsed_search_text=parsed.search_text,
            )
        )
    return candidates
