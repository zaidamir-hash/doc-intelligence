"""Transparent token-aware, structure-aware chunking for Lexis Phase 3."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from statistics import mean

import tiktoken

from pdf_processing import ExtractedPage, ExtractionResult
from settings import APP_SETTINGS


TOKENIZER_ENCODING = "cl100k_base"
CHUNKER_VERSION = "structure-token-v1"


@dataclass(frozen=True)
class ChunkingConfig:
    """The bounded settings that define one reproducible chunking run."""

    max_tokens: int = 300
    overlap_tokens: int = 50
    min_chunk_tokens: int = 40
    encoding_name: str = TOKENIZER_ENCODING
    version: str = CHUNKER_VERSION

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be positive")
        if self.min_chunk_tokens >= self.max_tokens:
            raise ValueError("min_chunk_tokens must be smaller than max_tokens")
        if not self.encoding_name.strip():
            raise ValueError("encoding_name cannot be empty")
        if not self.version.strip():
            raise ValueError("version cannot be empty")


PRODUCTION_CHUNK_CONFIG = ChunkingConfig(
    max_tokens=APP_SETTINGS.chunk_max_tokens,
    overlap_tokens=APP_SETTINGS.chunk_overlap_tokens,
    min_chunk_tokens=APP_SETTINGS.chunk_min_tokens,
    encoding_name=APP_SETTINGS.chunk_encoding_name,
)


@dataclass(frozen=True)
class StructuralUnit:
    """A paragraph, list item, sentence, or safe hard-split text unit."""

    text: str
    page_start: int
    page_end: int
    section_title: str | None
    boundary_kind: str


@dataclass(frozen=True)
class StructuredChunk:
    """A retrievable passage plus inspectable provenance and identity metadata."""

    content: str
    passage_text: str
    page_start: int
    page_end: int
    section_title: str | None
    token_count: int
    overlap_token_count: int
    content_hash: str
    chunking_version: str
    boundary_kinds: tuple[str, ...]


class Tokenizer:
    """Small wrapper keeping all tokenizer behavior explicit and testable."""

    def __init__(self, encoding_name: str = TOKENIZER_ENCODING) -> None:
        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)

    def count(self, text: str) -> int:
        return len(self.encode(text))


def _is_list_item(line: str) -> bool:
    return bool(re.match(r"^(?:[-*\uf0b7\u2022]|\d+[.)]|[A-Za-z][.)])\s+", line.strip()))


def _looks_like_heading(line: str, tokenizer: Tokenizer) -> bool:
    """Recognize only short, strong heading forms to limit false positives."""

    candidate = re.sub(r"\s+", " ", line).strip()
    if not candidate or len(candidate) > 120 or tokenizer.count(candidate) > 18:
        return False
    if re.match(r"^(?:mr|mrs|ms|dr|prof)\.", candidate, re.I):
        return False
    if ":" in candidate or re.match(r"^\d{2,}\s", candidate):
        return False
    if _is_list_item(candidate) or candidate.endswith((".", "!", "?", ";", ",")):
        return False
    if candidate.isdigit() or not any(character.isalpha() for character in candidate):
        return False
    if re.match(
        r"^(?:chapter\s+\d+|\d+(?:\.\d+){1,3}|[1-9])\s+\D",
        candidate,
        re.I,
    ):
        return True
    letters = [character for character in candidate if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z&/'’-]*", candidate)
    if 1 <= len(words) <= 12:
        significant = [word for word in words if word.casefold() not in {"a", "an", "and", "of", "the", "to"}]
        return bool(significant) and all(word[0].isupper() for word in significant)
    return False


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extract_structural_units(
    pages: tuple[ExtractedPage, ...], tokenizer: Tokenizer
) -> list[StructuralUnit]:
    """Turn cleaned pages into ordered structure-bearing text units."""

    units: list[StructuralUnit] = []
    current_section: str | None = None

    for page in pages:
        paragraphs = re.split(r"\n\s*\n", page.cleaned_text)
        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines:
                continue

            heading_lines: list[str] = []
            while lines and _looks_like_heading(lines[0], tokenizer):
                if heading_lines and re.match(
                    r"^(?:table|figure|source|note)\b", lines[0], re.I
                ):
                    break
                proposed = " ".join(heading_lines + [lines[0]])
                if len(proposed) > 120 or tokenizer.count(proposed) > 18:
                    break
                heading_lines.append(lines.pop(0))
            if heading_lines:
                current_section = " ".join(heading_lines)
                if not lines:
                    continue

            if all(_is_list_item(line) for line in lines):
                for line in lines:
                    units.append(
                        StructuralUnit(
                            text=line,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            section_title=current_section,
                            boundary_kind="list-item",
                        )
                    )
                continue

            units.append(
                StructuralUnit(
                    text="\n".join(lines),
                    page_start=page.page_number,
                    page_end=page.page_number,
                    section_title=current_section,
                    boundary_kind="paragraph",
                )
            )
    return units


def _canonical_hash(
    passage_text: str,
    page_start: int,
    page_end: int,
    section_title: str | None,
    version: str,
) -> str:
    payload = json.dumps(
        {
            "passage_text": passage_text,
            "page_start": page_start,
            "page_end": page_end,
            "section_title": section_title,
            "chunking_version": version,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _page_label(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"[Source page {page_start}]"
    return f"[Source pages {page_start}-{page_end}]"


def _render_chunk(
    passage_text: str,
    page_start: int,
    page_end: int,
    section_title: str | None,
    overlap_token_count: int,
    boundary_kinds: tuple[str, ...],
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> StructuredChunk:
    content_hash = _canonical_hash(
        passage_text, page_start, page_end, section_title, config.version
    )
    parts = [_page_label(page_start, page_end)]
    if section_title:
        parts.append(f"[Section: {section_title}]")
    parts.append(passage_text)
    content = "\n".join(parts)
    token_count = tokenizer.count(content)

    return StructuredChunk(
        content=content,
        passage_text=passage_text,
        page_start=page_start,
        page_end=page_end,
        section_title=section_title,
        token_count=token_count,
        overlap_token_count=overlap_token_count,
        content_hash=content_hash,
        chunking_version=config.version,
        boundary_kinds=boundary_kinds,
    )


def _join_unit_text(units: list[StructuralUnit]) -> str:
    parts: list[str] = []
    for unit in units:
        if not parts:
            parts.append(unit.text)
        elif unit.boundary_kind == "list-item" and units[0].boundary_kind == "list-item":
            parts.append("\n" + unit.text)
        else:
            parts.append("\n\n" + unit.text)
    return "".join(parts).strip()


def _chunk_from_units(
    units: list[StructuralUnit],
    config: ChunkingConfig,
    tokenizer: Tokenizer,
    overlap_token_count: int = 0,
) -> StructuredChunk:
    if not units:
        raise ValueError("cannot create a chunk from no units")
    return _render_chunk(
        passage_text=_join_unit_text(units),
        page_start=min(unit.page_start for unit in units),
        page_end=max(unit.page_end for unit in units),
        section_title=units[-1].section_title,
        overlap_token_count=overlap_token_count,
        boundary_kinds=tuple(dict.fromkeys(unit.boundary_kind for unit in units)),
        config=config,
        tokenizer=tokenizer,
    )


def _split_oversized_unit(
    unit: StructuralUnit,
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[StructuralUnit]:
    """Prefer sentence boundaries, then use a token-safe hard boundary."""

    single = _chunk_from_units([unit], config, tokenizer)
    if single.token_count <= config.max_tokens:
        return [unit]

    sentence_units = [
        replace(unit, text=sentence, boundary_kind="sentence")
        for sentence in _split_sentences(unit.text)
    ]
    if len(sentence_units) > 1:
        output: list[StructuralUnit] = []
        for sentence_unit in sentence_units:
            output.extend(_split_oversized_unit(sentence_unit, config, tokenizer))
        return output

    remaining_ids = tokenizer.encode(unit.text)
    output = []
    while remaining_ids:
        low = 1
        high = len(remaining_ids)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            candidate_text = tokenizer.decode(remaining_ids[:middle]).strip()
            if not candidate_text:
                low = middle + 1
                continue
            candidate = replace(
                unit, text=candidate_text, boundary_kind="hard-token-boundary"
            )
            if _chunk_from_units([candidate], config, tokenizer).token_count <= config.max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise ValueError("max_tokens is too small for required chunk metadata")
        text = tokenizer.decode(remaining_ids[:best]).strip()
        output.append(replace(unit, text=text, boundary_kind="hard-token-boundary"))
        remaining_ids = remaining_ids[best:]
    return output


def _prepare_units(
    extraction: ExtractionResult,
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[StructuralUnit]:
    prepared: list[StructuralUnit] = []
    for unit in extract_structural_units(extraction.pages, tokenizer):
        prepared.extend(_split_oversized_unit(unit, config, tokenizer))
    return [unit for unit in prepared if unit.text.strip()]


def _pack_base_chunks(
    units: list[StructuralUnit],
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[list[StructuralUnit]]:
    packed: list[list[StructuralUnit]] = []
    current: list[StructuralUnit] = []

    for unit in units:
        section_changed = bool(
            current and unit.section_title != current[-1].section_title
        )
        candidate_units = current + [unit]
        candidate_fits = (
            not section_changed
            and _chunk_from_units(candidate_units, config, tokenizer).token_count
            <= config.max_tokens - config.overlap_tokens
        )
        if current and not candidate_fits:
            packed.append(current)
            current = [unit]
        else:
            current = candidate_units
    if current:
        packed.append(current)

    for index in range(len(packed) - 1, 0, -1):
        current_chunk = _chunk_from_units(packed[index], config, tokenizer)
        if current_chunk.token_count >= config.min_chunk_tokens:
            continue
        if packed[index - 1][-1].section_title != packed[index][0].section_title:
            continue
        while len(packed[index - 1]) > 1:
            moved = packed[index - 1][-1]
            proposed_previous = packed[index - 1][:-1]
            proposed_current = [moved] + packed[index]
            if (
                _chunk_from_units(proposed_previous, config, tokenizer).token_count
                < config.min_chunk_tokens
            ):
                break
            if (
                _chunk_from_units(proposed_current, config, tokenizer).token_count
                > config.max_tokens - config.overlap_tokens
            ):
                break
            packed[index - 1] = proposed_previous
            packed[index] = proposed_current
            if (
                _chunk_from_units(packed[index], config, tokenizer).token_count
                >= config.min_chunk_tokens
            ):
                break
    return packed


def _with_overlap(
    base_chunks: list[list[StructuralUnit]],
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[StructuredChunk]:
    chunks: list[StructuredChunk] = []
    for index, base_units in enumerate(base_chunks):
        if index == 0 or config.overlap_tokens == 0:
            chunks.append(_chunk_from_units(base_units, config, tokenizer))
            continue

        previous_units = base_chunks[index - 1]
        if previous_units[-1].section_title != base_units[0].section_title:
            chunks.append(_chunk_from_units(base_units, config, tokenizer))
            continue

        previous_text = _join_unit_text(previous_units)
        previous_ids = tokenizer.encode(previous_text)
        requested = min(config.overlap_tokens, len(previous_ids))
        chosen_overlap = 0
        chosen_text = ""
        for overlap_size in range(requested, 0, -1):
            overlap_text = tokenizer.decode(previous_ids[-overlap_size:]).strip()
            if not overlap_text:
                continue
            overlap_unit = StructuralUnit(
                text=overlap_text,
                page_start=previous_units[-1].page_end,
                page_end=previous_units[-1].page_end,
                section_title=base_units[0].section_title,
                boundary_kind="token-overlap",
            )
            candidate = _chunk_from_units(
                [overlap_unit] + base_units,
                config,
                tokenizer,
                overlap_token_count=overlap_size,
            )
            if candidate.token_count <= config.max_tokens:
                chosen_overlap = overlap_size
                chosen_text = overlap_text
                break

        if chosen_overlap:
            overlap_unit = StructuralUnit(
                text=chosen_text,
                page_start=previous_units[-1].page_end,
                page_end=previous_units[-1].page_end,
                section_title=base_units[0].section_title,
                boundary_kind="token-overlap",
            )
            chunks.append(
                _chunk_from_units(
                    [overlap_unit] + base_units,
                    config,
                    tokenizer,
                    overlap_token_count=chosen_overlap,
                )
            )
        else:
            chunks.append(_chunk_from_units(base_units, config, tokenizer))
    return chunks


def chunk_extraction(
    extraction: ExtractionResult,
    config: ChunkingConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[StructuredChunk]:
    """Create finite, non-empty token-bounded chunks from cleaned PDF pages."""

    config = config or ChunkingConfig()
    config.validate()
    tokenizer = tokenizer or Tokenizer(config.encoding_name)
    units = _prepare_units(extraction, config, tokenizer)
    if not units:
        return []
    base_chunks = _pack_base_chunks(units, config, tokenizer)
    chunks = _with_overlap(base_chunks, config, tokenizer)
    if any(not chunk.passage_text.strip() for chunk in chunks):
        raise RuntimeError("chunker produced an empty chunk")
    if any(chunk.token_count > config.max_tokens for chunk in chunks):
        raise RuntimeError("chunker exceeded max_tokens")
    unique_chunks: list[StructuredChunk] = []
    seen_hashes: set[str] = set()
    for chunk in chunks:
        if chunk.content_hash in seen_hashes:
            continue
        seen_hashes.add(chunk.content_hash)
        unique_chunks.append(chunk)
    return unique_chunks


def _token_jaccard(first: str, second: str, tokenizer: Tokenizer) -> float:
    first_tokens = set(tokenizer.encode(first))
    second_tokens = set(tokenizer.encode(second))
    union = first_tokens.union(second_tokens)
    return 0.0 if not union else len(first_tokens.intersection(second_tokens)) / len(union)


def summarize_chunks(
    chunks: list[StructuredChunk], tokenizer: Tokenizer | None = None
) -> dict[str, object]:
    """Return size, provenance, structure, identity, and overlap diagnostics."""

    tokenizer = tokenizer or Tokenizer()
    token_counts = [chunk.token_count for chunk in chunks]
    adjacent_similarities = [
        _token_jaccard(first.passage_text, second.passage_text, tokenizer)
        for first, second in zip(chunks, chunks[1:])
    ]
    return {
        "chunk_count": len(chunks),
        "token_count": {
            "minimum": min(token_counts) if token_counts else 0,
            "maximum": max(token_counts) if token_counts else 0,
            "mean": mean(token_counts) if token_counts else 0.0,
        },
        "cross_page_chunks": sum(
            chunk.page_start != chunk.page_end for chunk in chunks
        ),
        "chunks_with_heading": sum(bool(chunk.section_title) for chunk in chunks),
        "exact_duplicate_hashes": len(chunks)
        - len({chunk.content_hash for chunk in chunks}),
        "adjacent_token_jaccard": {
            "mean": mean(adjacent_similarities) if adjacent_similarities else 0.0,
            "maximum": max(adjacent_similarities) if adjacent_similarities else 0.0,
            "near_duplicate_pairs_at_0_8": sum(
                similarity >= 0.8 for similarity in adjacent_similarities
            ),
        },
        "actual_overlap_tokens": {
            "mean": mean(chunk.overlap_token_count for chunk in chunks)
            if chunks
            else 0.0,
            "maximum": max((chunk.overlap_token_count for chunk in chunks), default=0),
        },
    }
