"""Page-aware PDF extraction and conservative text cleaning.

Phase 2 deliberately keeps the existing character-based chunking algorithm.
Its responsibility is to make the input to that algorithm cleaner and to keep
every stored chunk traceable to the PDF page it came from.
"""

from __future__ import annotations

import io
import math
import re
from collections import Counter
from dataclasses import dataclass

from pypdf import PdfReader


EXTRACTION_VERSION = "page-aware-v2"
LOW_TEXT_CHARACTER_THRESHOLD = 40
FURNITURE_POSITION_LINES = 2
FURNITURE_MINIMUM_PAGES = 3
FURNITURE_PAGE_FRACTION = 0.25
PAGE_LABEL_TEMPLATE = "[Source page {page_number}]"
_SUSPICIOUS_WORD_BREAK = re.compile(r"(?m)\b[A-Z][A-Za-z]{0,2}\n[a-z]{3,}")


@dataclass(frozen=True)
class ExtractionWarning:
    """A visible extraction problem associated with one PDF page."""

    code: str
    message: str


@dataclass(frozen=True)
class ExtractedPage:
    """Raw and cleaned text for one original, one-based PDF page."""

    page_number: int
    raw_text: str
    cleaned_text: str
    warnings: tuple[ExtractionWarning, ...]
    removed_headers: tuple[str, ...] = ()
    removed_footers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractionResult:
    """All page-level extraction output and document-wide diagnostics."""

    pages: tuple[ExtractedPage, ...]
    repeated_headers: tuple[str, ...]
    repeated_footers: tuple[str, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def warning_count(self) -> int:
        return sum(len(page.warnings) for page in self.pages)


@dataclass(frozen=True)
class PageChunk:
    """A current-generation character chunk with explicit page provenance."""

    content: str
    page_start: int
    page_end: int


def _normalize_newlines_and_spaces(text: str) -> str:
    """Normalize invisible spacing differences without removing line structure."""

    normalized = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\u2007", " ")
        .replace("\u202f", " ")
    )
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)


def _extraction_quality_penalty(text: str) -> int:
    """Count strong signs that PDF positioning damaged readable words."""

    replacement_characters = text.count("\ufffd")
    split_words = len(_SUSPICIOUS_WORD_BREAK.findall(text))
    return replacement_characters * 10 + split_words


def _extract_best_page_text(page: object) -> str:
    """Use layout extraction only when it measurably repairs damaged text."""

    plain_text = page.extract_text() or ""
    plain_penalty = _extraction_quality_penalty(plain_text)
    if plain_penalty == 0:
        return plain_text
    try:
        layout_text = page.extract_text(extraction_mode="layout") or ""
    except Exception:
        return plain_text
    if _extraction_quality_penalty(layout_text) < plain_penalty:
        return layout_text
    return plain_text


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _furniture_key(line: str) -> str:
    """Create a cautious comparison key for recurring page-edge text."""

    normalized = re.sub(r"\s+", " ", line).strip().casefold()
    if re.fullmatch(r"(?:page\s*)?\d+", normalized):
        return "<page-number>"
    return normalized


def _repeated_furniture(
    raw_pages: list[str], *, from_start: bool
) -> tuple[set[str], tuple[str, ...]]:
    """Find lines recurring near the top or bottom of most text-bearing pages."""

    page_candidates: list[list[tuple[str, str]]] = []
    for raw_text in raw_pages:
        lines = _nonempty_lines(_normalize_newlines_and_spaces(raw_text))
        edge_lines = (
            lines[:FURNITURE_POSITION_LINES]
            if from_start
            else lines[-FURNITURE_POSITION_LINES:]
        )
        page_candidates.append(
            [(_furniture_key(line), line) for line in edge_lines if len(line) <= 120]
        )

    eligible_pages = sum(bool(candidates) for candidates in page_candidates)
    if eligible_pages < FURNITURE_MINIMUM_PAGES:
        return set(), ()

    minimum_occurrences = max(
        FURNITURE_MINIMUM_PAGES,
        math.ceil(eligible_pages * FURNITURE_PAGE_FRACTION),
    )
    counts = Counter(
        key
        for candidates in page_candidates
        for key in {candidate_key for candidate_key, _ in candidates}
        if key
    )
    repeated_keys = {
        key for key, count in counts.items() if count >= minimum_occurrences
    }

    representatives: dict[str, str] = {}
    for candidates in page_candidates:
        for key, original in candidates:
            if key in repeated_keys and key not in representatives:
                representatives[key] = original
    return repeated_keys, tuple(representatives[key] for key in sorted(repeated_keys))


def _looks_structured(line: str) -> bool:
    """Protect list items, labels, and table-like rows from line joining."""

    stripped = line.strip()
    return bool(
        re.match(r"^(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+", stripped)
        or "\t" in line
        or re.search(r"\s{3,}", line)
        or stripped.endswith(":")
    )


def _should_join_wrapped_lines(previous: str, current: str) -> bool:
    """Join only strong prose-wrap cases to avoid damaging tables and lists."""

    if not previous or not current:
        return False
    if _looks_structured(previous) or _looks_structured(current):
        return False
    if previous.endswith((".", "!", "?", ":", ";")):
        return False
    if previous.endswith("-") and current[0].islower():
        return True
    return current[0].islower() and len(previous) >= 35


def _clean_lines(lines: list[str]) -> str:
    """Normalize whitespace and conservatively repair prose line wrapping."""

    paragraphs: list[list[str]] = []
    current_paragraph: list[str] = []

    for raw_line in lines:
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []
            continue

        if current_paragraph and _should_join_wrapped_lines(
            current_paragraph[-1], line
        ):
            previous = current_paragraph.pop()
            if previous.endswith("-") and line[0].islower():
                current_paragraph.append(previous[:-1] + line)
            else:
                current_paragraph.append(previous + " " + line)
        else:
            current_paragraph.append(line)

    if current_paragraph:
        paragraphs.append(current_paragraph)

    cleaned_paragraphs = ["\n".join(paragraph) for paragraph in paragraphs]
    retained_paragraphs: list[str] = []
    seen_long_paragraphs: set[str] = set()
    for paragraph in cleaned_paragraphs:
        comparison_key = re.sub(r"\s+", " ", paragraph).strip()
        if len(comparison_key) >= 80:
            if comparison_key in seen_long_paragraphs:
                continue
            seen_long_paragraphs.add(comparison_key)
        retained_paragraphs.append(paragraph)
    return "\n\n".join(retained_paragraphs).strip()


def _remove_page_furniture(
    text: str,
    repeated_header_keys: set[str],
    repeated_footer_keys: set[str],
) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    """Remove recurring text only when it occurs at the corresponding page edge."""

    lines = _normalize_newlines_and_spaces(text).splitlines()
    nonempty_positions = [index for index, line in enumerate(lines) if line.strip()]
    header_positions = set(nonempty_positions[:FURNITURE_POSITION_LINES])
    footer_positions = set(nonempty_positions[-FURNITURE_POSITION_LINES:])
    removed_headers: list[str] = []
    removed_footers: list[str] = []
    retained: list[str] = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        key = _furniture_key(stripped) if stripped else ""
        if index in header_positions and key in repeated_header_keys:
            removed_headers.append(stripped)
        elif index in footer_positions and key in repeated_footer_keys:
            removed_footers.append(stripped)
        else:
            retained.append(line)

    return retained, tuple(removed_headers), tuple(removed_footers)


def process_extracted_pages(
    raw_pages: list[str], extraction_errors: dict[int, str] | None = None
) -> ExtractionResult:
    """Clean already-extracted pages; separated for focused deterministic tests."""

    extraction_errors = extraction_errors or {}
    header_keys, repeated_headers = _repeated_furniture(
        raw_pages, from_start=True
    )
    footer_keys, repeated_footers = _repeated_furniture(
        raw_pages, from_start=False
    )

    processed_pages: list[ExtractedPage] = []
    for page_number, raw_text in enumerate(raw_pages, start=1):
        page_header_keys = set() if page_number == 1 else header_keys
        page_footer_keys = set() if page_number == 1 else footer_keys
        retained_lines, removed_headers, removed_footers = _remove_page_furniture(
            raw_text, page_header_keys, page_footer_keys
        )
        cleaned_text = _clean_lines(retained_lines)
        warnings: list[ExtractionWarning] = []

        if page_number in extraction_errors:
            warnings.append(
                ExtractionWarning(
                    code="extraction-error",
                    message=f"Text extraction failed: {extraction_errors[page_number]}",
                )
            )
        visible_character_count = len(re.sub(r"\s+", "", cleaned_text))
        if visible_character_count == 0:
            warnings.append(
                ExtractionWarning(
                    code="empty-page",
                    message="No extractable text remained after cleaning.",
                )
            )
        elif visible_character_count < LOW_TEXT_CHARACTER_THRESHOLD:
            warnings.append(
                ExtractionWarning(
                    code="low-text-page",
                    message=(
                        f"Only {visible_character_count} non-whitespace characters "
                        "were extracted."
                    ),
                )
            )

        processed_pages.append(
            ExtractedPage(
                page_number=page_number,
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                warnings=tuple(warnings),
                removed_headers=removed_headers,
                removed_footers=removed_footers,
            )
        )

    return ExtractionResult(
        pages=tuple(processed_pages),
        repeated_headers=repeated_headers,
        repeated_footers=repeated_footers,
    )


def extract_pdf(contents: bytes) -> ExtractionResult:
    """Extract and clean a PDF while retaining original one-based page numbers."""

    reader = PdfReader(io.BytesIO(contents))
    raw_pages: list[str] = []
    extraction_errors: dict[int, str] = {}

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_pages.append(_extract_best_page_text(page))
        except Exception as error:  # one broken page should remain visible
            raw_pages.append("")
            extraction_errors[page_number] = str(error)

    return process_extracted_pages(raw_pages, extraction_errors)


def find_break_point(text: str, start: int, end: int) -> int:
    """Find the same paragraph/sentence/line boundary used by the baseline."""

    search_window = 200
    search_start = max(start, end - search_window)
    chunk = text[search_start:end]

    paragraph_break = chunk.rfind("\n\n")
    if paragraph_break != -1:
        return search_start + paragraph_break + 2

    for punctuation in [". ", "! ", "? "]:
        sentence_break = chunk.rfind(punctuation)
        if sentence_break != -1:
            return search_start + sentence_break + len(punctuation)

    line_break = chunk.rfind("\n")
    if line_break != -1:
        return search_start + line_break + 1
    return end


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Character chunking retained unchanged until the dedicated Phase 3 work."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = find_break_point(text, start, end)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def chunk_extracted_pages(
    extraction: ExtractionResult,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[PageChunk]:
    """Chunk each page independently and add visible provenance to every chunk."""

    chunks: list[PageChunk] = []
    for page in extraction.pages:
        for chunk in chunk_text(page.cleaned_text, chunk_size, overlap):
            page_label = PAGE_LABEL_TEMPLATE.format(page_number=page.page_number)
            chunks.append(
                PageChunk(
                    content=f"{page_label}\n{chunk}",
                    page_start=page.page_number,
                    page_end=page.page_number,
                )
            )
    return chunks


def summarize_extraction(extraction: ExtractionResult) -> dict[str, object]:
    """Return compact diagnostics suitable for API responses and reports."""

    pages_with_warnings = [
        {
            "page_number": page.page_number,
            "warnings": [
                {"code": warning.code, "message": warning.message}
                for warning in page.warnings
            ],
        }
        for page in extraction.pages
        if page.warnings
    ]
    return {
        "page_count": extraction.page_count,
        "warning_count": extraction.warning_count,
        "pages_with_warnings": pages_with_warnings,
        "repeated_headers": list(extraction.repeated_headers),
        "repeated_footers": list(extraction.repeated_footers),
    }
