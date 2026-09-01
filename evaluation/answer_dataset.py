"""Load and validate the bounded Phase 11 answer-evaluation dataset."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


SUPPORTED_SCHEMA_VERSION = 1
ALLOWED_SPLITS = {"development", "validation", "test"}
ALLOWED_CASE_TYPES = {
    "answerable",
    "unanswerable",
    "partially_answerable",
    "conflicting_evidence",
}
ALLOWED_RUN_MODES = {"pipeline", "synthetic_context"}
ALLOWED_STATUSES = {
    "answered",
    "partially_answered",
    "insufficient_evidence",
}


class AnswerDatasetValidationError(ValueError):
    """Raised when Phase 11 gold labels are incomplete or inconsistent."""


@dataclass(frozen=True)
class ClaimExpectation:
    """One human-readable fact expected in a correct answer."""

    claim_id: str
    description: str
    required: bool
    term_groups: tuple[tuple[str, ...], ...]
    evidence_hashes: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticEvidence:
    """One fixed passage for a controlled partial/conflict evaluation case."""

    content_hash: str
    page_start: int
    page_end: int
    section_title: str
    passage_text: str


@dataclass(frozen=True)
class AnswerEvaluationCase:
    """Question, gold answer facts, and expected evidence for one case."""

    case_id: str
    run_mode: Literal["pipeline", "synthetic_context"]
    case_type: str
    filename: str
    question: str
    document_content_hash: str
    reference_answer: str | None
    expected_statuses: tuple[str, ...]
    expected_evidence_hashes: tuple[str, ...]
    claims: tuple[ClaimExpectation, ...]
    synthetic_evidence: tuple[SyntheticEvidence, ...]
    tags: tuple[str, ...]
    notes: str | None


@dataclass(frozen=True)
class AnswerEvaluationDataset:
    """Versioned metadata and bounded Phase 11 cases."""

    schema_version: int
    name: str
    description: str
    split: str
    source_retrieval_dataset: str
    label_policy: str
    cases: tuple[AnswerEvaluationCase, ...]
    path: Path
    sha256: str


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerDatasetValidationError(
            f"{field_name} must be a non-empty string"
        )
    return value.strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnswerDatasetValidationError(f"{field_name} must be a string")
    return value.strip() or None


def _sha256(value: Any, field_name: str) -> str:
    parsed = _non_empty_string(value, field_name)
    if not re.fullmatch(r"[0-9a-f]{64}", parsed):
        raise AnswerDatasetValidationError(
            f"{field_name} must be a lowercase SHA-256 hash"
        )
    return parsed


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AnswerDatasetValidationError(f"{field_name} must be a list")
    parsed = tuple(_non_empty_string(item, field_name) for item in value)
    if len(parsed) != len(set(parsed)):
        raise AnswerDatasetValidationError(f"{field_name} contains duplicates")
    return parsed


def _hash_list(value: Any, field_name: str) -> tuple[str, ...]:
    parsed = tuple(_sha256(item, field_name) for item in value)
    if len(parsed) != len(set(parsed)):
        raise AnswerDatasetValidationError(f"{field_name} contains duplicates")
    return parsed


def _parse_claim(raw: Any, case_id: str) -> ClaimExpectation:
    if not isinstance(raw, dict):
        raise AnswerDatasetValidationError(
            f"Case {case_id}: every expected claim must be an object"
        )
    claim_id = _non_empty_string(
        raw.get("id"), f"Case {case_id}: expected claim id"
    )
    description = _non_empty_string(
        raw.get("description"), f"Case {case_id}/{claim_id}: description"
    )
    required = raw.get("required", True)
    if not isinstance(required, bool):
        raise AnswerDatasetValidationError(
            f"Case {case_id}/{claim_id}: required must be boolean"
        )
    raw_groups = raw.get("term_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise AnswerDatasetValidationError(
            f"Case {case_id}/{claim_id}: term_groups must be a non-empty list"
        )
    groups = tuple(
        _string_list(group, f"Case {case_id}/{claim_id}: term group")
        for group in raw_groups
    )
    evidence_hashes = _hash_list(
        raw.get("evidence_hashes", []),
        f"Case {case_id}/{claim_id}: evidence_hashes",
    )
    return ClaimExpectation(
        claim_id=claim_id,
        description=description,
        required=required,
        term_groups=groups,
        evidence_hashes=evidence_hashes,
    )


def _parse_synthetic_evidence(
    raw: Any,
    case_id: str,
) -> SyntheticEvidence:
    if not isinstance(raw, dict):
        raise AnswerDatasetValidationError(
            f"Case {case_id}: every synthetic evidence item must be an object"
        )
    page_start = raw.get("page_start")
    page_end = raw.get("page_end", page_start)
    if not isinstance(page_start, int) or page_start <= 0:
        raise AnswerDatasetValidationError(
            f"Case {case_id}: synthetic page_start must be positive"
        )
    if not isinstance(page_end, int) or page_end < page_start:
        raise AnswerDatasetValidationError(
            f"Case {case_id}: synthetic page_end must be >= page_start"
        )
    return SyntheticEvidence(
        content_hash=_sha256(
            raw.get("content_hash"),
            f"Case {case_id}: synthetic content_hash",
        ),
        page_start=page_start,
        page_end=page_end,
        section_title=_non_empty_string(
            raw.get("section_title"),
            f"Case {case_id}: synthetic section_title",
        ),
        passage_text=_non_empty_string(
            raw.get("passage_text"),
            f"Case {case_id}: synthetic passage_text",
        ),
    )


def _parse_case(raw: Any) -> AnswerEvaluationCase:
    if not isinstance(raw, dict):
        raise AnswerDatasetValidationError("Every case must be an object")
    case_id = _non_empty_string(raw.get("id"), "case id")
    run_mode = _non_empty_string(
        raw.get("run_mode"), f"Case {case_id}: run_mode"
    )
    if run_mode not in ALLOWED_RUN_MODES:
        raise AnswerDatasetValidationError(
            f"Case {case_id}: run_mode must be one of {sorted(ALLOWED_RUN_MODES)}"
        )
    case_type = _non_empty_string(
        raw.get("case_type"), f"Case {case_id}: case_type"
    )
    if case_type not in ALLOWED_CASE_TYPES:
        raise AnswerDatasetValidationError(
            f"Case {case_id}: case_type must be one of "
            f"{sorted(ALLOWED_CASE_TYPES)}"
        )
    filename = _non_empty_string(
        raw.get("filename"), f"Case {case_id}: filename"
    )
    document_content_hash = _sha256(
        raw.get("document_content_hash"),
        f"Case {case_id}: document_content_hash",
    )
    statuses = _string_list(
        raw.get("expected_statuses"),
        f"Case {case_id}: expected_statuses",
    )
    if not statuses or any(status not in ALLOWED_STATUSES for status in statuses):
        raise AnswerDatasetValidationError(
            f"Case {case_id}: expected_statuses contains an unsupported status"
        )
    evidence_hashes = _hash_list(
        raw.get("expected_evidence_hashes", []),
        f"Case {case_id}: expected_evidence_hashes",
    )
    raw_claims = raw.get("expected_claims", [])
    if not isinstance(raw_claims, list):
        raise AnswerDatasetValidationError(
            f"Case {case_id}: expected_claims must be a list"
        )
    claims = tuple(_parse_claim(item, case_id) for item in raw_claims)
    claim_ids = [claim.claim_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise AnswerDatasetValidationError(
            f"Case {case_id}: expected claim IDs must be unique"
        )
    raw_synthetic = raw.get("synthetic_evidence", [])
    if not isinstance(raw_synthetic, list):
        raise AnswerDatasetValidationError(
            f"Case {case_id}: synthetic_evidence must be a list"
        )
    synthetic_evidence = tuple(
        _parse_synthetic_evidence(item, case_id) for item in raw_synthetic
    )
    if run_mode == "pipeline" and synthetic_evidence:
        raise AnswerDatasetValidationError(
            f"Case {case_id}: pipeline cases cannot contain synthetic evidence"
        )
    if run_mode == "synthetic_context" and not synthetic_evidence:
        raise AnswerDatasetValidationError(
            f"Case {case_id}: synthetic_context requires evidence"
        )
    if case_type == "unanswerable":
        if claims or evidence_hashes:
            raise AnswerDatasetValidationError(
                f"Case {case_id}: unanswerable cases cannot require facts/evidence"
            )
        if statuses != ("insufficient_evidence",):
            raise AnswerDatasetValidationError(
                f"Case {case_id}: unanswerable case must require refusal"
            )
    else:
        if not any(claim.required for claim in claims):
            raise AnswerDatasetValidationError(
                f"Case {case_id}: answerable cases require a required claim"
            )
        claim_hashes = {
            content_hash for claim in claims for content_hash in claim.evidence_hashes
        }
        if not claim_hashes <= set(evidence_hashes):
            raise AnswerDatasetValidationError(
                f"Case {case_id}: claim evidence must be included in case evidence"
            )
    return AnswerEvaluationCase(
        case_id=case_id,
        run_mode=run_mode,  # type: ignore[arg-type]
        case_type=case_type,
        filename=filename,
        question=_non_empty_string(
            raw.get("question"), f"Case {case_id}: question"
        ),
        document_content_hash=document_content_hash,
        reference_answer=_optional_string(
            raw.get("reference_answer"), f"Case {case_id}: reference_answer"
        ),
        expected_statuses=statuses,
        expected_evidence_hashes=evidence_hashes,
        claims=claims,
        synthetic_evidence=synthetic_evidence,
        tags=_string_list(raw.get("tags", []), f"Case {case_id}: tags"),
        notes=_optional_string(raw.get("notes"), f"Case {case_id}: notes"),
    )


def load_answer_dataset(path: Path) -> AnswerEvaluationDataset:
    """Read a Phase 11 dataset and reject invalid gold labels."""

    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except FileNotFoundError as error:
        raise AnswerDatasetValidationError(f"Dataset not found: {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnswerDatasetValidationError(
            f"Dataset is not valid UTF-8 JSON: {path} ({error})"
        ) from error
    if not isinstance(raw, dict):
        raise AnswerDatasetValidationError("Dataset root must be an object")
    if raw.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise AnswerDatasetValidationError(
            f"Unsupported schema_version {raw.get('schema_version')!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )
    split = _non_empty_string(raw.get("split"), "dataset split")
    if split not in ALLOWED_SPLITS:
        raise AnswerDatasetValidationError(
            f"dataset split must be one of {sorted(ALLOWED_SPLITS)}"
        )
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise AnswerDatasetValidationError("cases must be a non-empty list")
    cases = tuple(_parse_case(item) for item in raw_cases)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise AnswerDatasetValidationError("case IDs must be unique")
    return AnswerEvaluationDataset(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        name=_non_empty_string(raw.get("name"), "dataset name"),
        description=_non_empty_string(
            raw.get("description"), "dataset description"
        ),
        split=split,
        source_retrieval_dataset=_non_empty_string(
            raw.get("source_retrieval_dataset"), "source_retrieval_dataset"
        ),
        label_policy=_non_empty_string(raw.get("label_policy"), "label_policy"),
        cases=cases,
        path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
