"""Load and validate versioned retrieval-evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_SCHEMA_VERSION = 1
ALLOWED_SPLITS = {"development", "validation", "test"}


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset is incomplete or inconsistent."""


@dataclass(frozen=True)
class EvaluationCase:
    """One question and its human-labelled relevant chunks."""

    case_id: str
    filename: str
    question: str
    answerable: bool
    relevant_chunk_indices: tuple[int, ...]
    relevance_grades: dict[int, int]
    reference_answer: str | None
    notes: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationDataset:
    """Metadata and cases loaded from one versioned JSON file."""

    schema_version: int
    name: str
    description: str
    split: str
    cases: tuple[EvaluationCase, ...]


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _parse_chunk_indices(value: Any, case_id: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise DatasetValidationError(
            f"Case {case_id}: relevant_chunk_indices must be a list"
        )
    if any(not isinstance(index, int) or index < 0 for index in value):
        raise DatasetValidationError(
            f"Case {case_id}: chunk indices must be non-negative integers"
        )
    if len(value) != len(set(value)):
        raise DatasetValidationError(
            f"Case {case_id}: relevant_chunk_indices contains duplicates"
        )
    return tuple(value)


def _parse_relevance_grades(
    value: Any,
    case_id: str,
    relevant_chunk_indices: tuple[int, ...],
) -> dict[int, int]:
    if value is None:
        return {index: 1 for index in relevant_chunk_indices}
    if not isinstance(value, dict):
        raise DatasetValidationError(
            f"Case {case_id}: relevance_grades must be an object"
        )

    grades: dict[int, int] = {}
    for raw_index, grade in value.items():
        try:
            chunk_index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(
                f"Case {case_id}: relevance grade keys must be chunk indices"
            ) from error
        if chunk_index not in relevant_chunk_indices:
            raise DatasetValidationError(
                f"Case {case_id}: grade supplied for unlabelled chunk {chunk_index}"
            )
        if not isinstance(grade, int) or grade <= 0:
            raise DatasetValidationError(
                f"Case {case_id}: relevance grades must be positive integers"
            )
        grades[chunk_index] = grade

    missing = set(relevant_chunk_indices) - set(grades)
    if missing:
        raise DatasetValidationError(
            f"Case {case_id}: missing relevance grades for chunks {sorted(missing)}"
        )
    return grades


def _parse_optional_string(value: Any, field_name: str, case_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetValidationError(f"Case {case_id}: {field_name} must be a string")
    return value.strip() or None


def _parse_case(raw_case: Any) -> EvaluationCase:
    if not isinstance(raw_case, dict):
        raise DatasetValidationError("Every case must be a JSON object")

    case_id = _require_non_empty_string(raw_case.get("id"), "case id")
    filename = _require_non_empty_string(
        raw_case.get("filename"), f"Case {case_id}: filename"
    )
    question = _require_non_empty_string(
        raw_case.get("question"), f"Case {case_id}: question"
    )
    answerable = raw_case.get("answerable")
    if not isinstance(answerable, bool):
        raise DatasetValidationError(f"Case {case_id}: answerable must be a boolean")

    relevant_indices = _parse_chunk_indices(
        raw_case.get("relevant_chunk_indices"), case_id
    )
    if answerable and not relevant_indices:
        raise DatasetValidationError(
            f"Case {case_id}: answerable cases require at least one relevant chunk"
        )
    if not answerable and relevant_indices:
        raise DatasetValidationError(
            f"Case {case_id}: unanswerable cases cannot have relevant chunks"
        )

    grades = _parse_relevance_grades(
        raw_case.get("relevance_grades"), case_id, relevant_indices
    )
    raw_tags = raw_case.get("tags", [])
    if not isinstance(raw_tags, list) or any(
        not isinstance(tag, str) or not tag.strip() for tag in raw_tags
    ):
        raise DatasetValidationError(
            f"Case {case_id}: tags must be a list of non-empty strings"
        )

    return EvaluationCase(
        case_id=case_id,
        filename=filename,
        question=question,
        answerable=answerable,
        relevant_chunk_indices=relevant_indices,
        relevance_grades=grades,
        reference_answer=_parse_optional_string(
            raw_case.get("reference_answer"), "reference_answer", case_id
        ),
        notes=_parse_optional_string(raw_case.get("notes"), "notes", case_id),
        tags=tuple(tag.strip() for tag in raw_tags),
    )


def load_dataset(path: Path) -> EvaluationDataset:
    """Read a JSON dataset and fail early when its labels are invalid."""

    try:
        raw_dataset = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetValidationError(f"Dataset not found: {path}") from error
    except json.JSONDecodeError as error:
        raise DatasetValidationError(
            f"Dataset is not valid JSON: {path} ({error})"
        ) from error

    if not isinstance(raw_dataset, dict):
        raise DatasetValidationError("Dataset root must be a JSON object")

    schema_version = raw_dataset.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise DatasetValidationError(
            f"Unsupported schema_version {schema_version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )

    name = _require_non_empty_string(raw_dataset.get("name"), "dataset name")
    description = _require_non_empty_string(
        raw_dataset.get("description"), "dataset description"
    )
    split = _require_non_empty_string(raw_dataset.get("split"), "dataset split")
    if split not in ALLOWED_SPLITS:
        raise DatasetValidationError(
            f"dataset split must be one of {sorted(ALLOWED_SPLITS)}"
        )

    raw_cases = raw_dataset.get("cases")
    if not isinstance(raw_cases, list):
        raise DatasetValidationError("cases must be a list")
    cases = tuple(_parse_case(raw_case) for raw_case in raw_cases)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise DatasetValidationError("case ids must be unique")

    return EvaluationDataset(
        schema_version=schema_version,
        name=name,
        description=description,
        split=split,
        cases=cases,
    )
