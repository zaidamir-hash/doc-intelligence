"""Run and report Phase 11 answer-quality evaluation separately from retrieval."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from sqlalchemy.orm import Session

from grounded_generation import (
    Citation,
    DEFAULT_ANSWER_MODEL,
    EvidenceItem,
    GeneratedAnswer,
    GenerationContext,
    GenerationUsage,
    GroundedAnswer,
    GroundedClaim,
    generate_grounded_answer,
    validate_generated_answer,
)
from models import DOCUMENT_STATUS_READY, Document, DocumentChunk
from query import (
    DEFAULT_DENSE_CANDIDATE_K,
    DEFAULT_FINAL_EVIDENCE_K,
    DEFAULT_LEXICAL_CANDIDATE_K,
    DEFAULT_RELEVANCE_CUTOFF,
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANK_K,
    DEFAULT_RRF_K,
    GroundedQueryResult,
    answer_document_question,
)
from query_expansion import DEFAULT_EXPANSION_MODEL
from reranking import DEFAULT_RERANKER_MODEL

from .answer_dataset import (
    AnswerEvaluationCase,
    AnswerEvaluationDataset,
    ClaimExpectation,
)


GRADER_VERSION = "phase11-deterministic-gold-v1"
MODEL_JUDGE = None
EMBEDDING_MODEL = "text-embedding-3-small"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _normalized_text(value: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(value.casefold()))


def claim_matches_expectation(
    claim_text: str,
    expectation: ClaimExpectation,
) -> bool:
    """Match a generated claim to fixed, inspectable gold term groups."""

    normalized_claim = f" {_normalized_text(claim_text)} "
    return all(
        any(
            f" {_normalized_text(alternative)} " in normalized_claim
            for alternative in group
        )
        for group in expectation.term_groups
    )


def _faithfulness_error(
    claim: GroundedClaim,
    context: GenerationContext,
) -> str | None:
    """Reuse Phase 10's claim/quote validator for one displayed claim."""

    try:
        validate_generated_answer(
            GeneratedAnswer(
                status="answered",
                claims=(claim,),
                insufficient_reason=None,
            ),
            context,
        )
    except ValueError as error:
        return str(error)
    return None


def grade_answer(
    case: AnswerEvaluationCase,
    answer: GroundedAnswer,
    context: GenerationContext,
) -> dict[str, Any]:
    """Calculate separate deterministic dimensions from fixed gold labels."""

    expectation_matches: dict[str, list[int]] = {}
    claim_matches: dict[int, list[str]] = {index: [] for index in range(len(answer.claims))}
    for expectation in case.claims:
        matching_indices = [
            index
            for index, claim in enumerate(answer.claims)
            if claim_matches_expectation(claim.text, expectation)
        ]
        expectation_matches[expectation.claim_id] = matching_indices
        for index in matching_indices:
            claim_matches[index].append(expectation.claim_id)

    required = [expectation for expectation in case.claims if expectation.required]
    matched_required = [
        expectation
        for expectation in required
        if expectation_matches[expectation.claim_id]
    ]
    if case.case_type == "unanswerable":
        completeness: float | None = None
        correctness = 1.0 if not answer.claims else 0.0
    else:
        completeness = len(matched_required) / len(required)
        correctness = (
            sum(bool(claim_matches[index]) for index in claim_matches)
            / len(answer.claims)
            if answer.claims
            else 0.0
        )

    faithfulness_errors = [
        _faithfulness_error(claim, context) for claim in answer.claims
    ]
    supported_claim_count = sum(error is None for error in faithfulness_errors)
    unsupported_claim_count = len(answer.claims) - supported_claim_count
    unsupported_claim_rate = (
        unsupported_claim_count / len(answer.claims) if answer.claims else 0.0
    )
    faithfulness = (
        supported_claim_count / len(answer.claims)
        if answer.claims
        else (1.0 if case.case_type == "unanswerable" else 0.0)
    )

    evidence_by_source = {item.source_id: item for item in context.evidence}
    citations_by_source = {
        citation.source_id: citation for citation in answer.citations
    }
    citation_checks: list[dict[str, Any]] = []
    for claim_index, claim in enumerate(answer.claims):
        matching_expectations = [
            expectation
            for expectation in case.claims
            if expectation.claim_id in claim_matches[claim_index]
        ]
        for source_id, quote in zip(claim.source_ids, claim.supporting_quotes):
            evidence = evidence_by_source.get(source_id)
            citation = citations_by_source.get(source_id)
            mapped = bool(
                evidence is not None
                and citation is not None
                and citation.chunk_id == evidence.chunk_id
                and citation.chunk_content_hash == evidence.chunk_content_hash
                and claim.text in citation.supported_claims
                and quote in citation.supporting_quotes
            )
            allowed_hashes = {
                content_hash
                for expectation in matching_expectations
                for content_hash in expectation.evidence_hashes
            }
            gold_evidence_match = (
                True
                if not allowed_hashes
                else bool(evidence and evidence.chunk_content_hash in allowed_hashes)
            )
            citation_checks.append(
                {
                    "claim_index": claim_index,
                    "source_id": source_id,
                    "quote": quote,
                    "maps_to_generation_evidence": mapped,
                    "gold_evidence_match": gold_evidence_match,
                    "accurate": (
                        mapped
                        and faithfulness_errors[claim_index] is None
                        and gold_evidence_match
                    ),
                }
            )
    citation_accuracy = (
        sum(check["accurate"] for check in citation_checks) / len(citation_checks)
        if citation_checks
        else (
            1.0
            if case.case_type == "unanswerable" and not answer.citations
            else 0.0
        )
    )

    status_matches = answer.status in case.expected_statuses
    if case.case_type == "unanswerable":
        refusal_behavior_matches = (
            answer.status == "insufficient_evidence"
            and not answer.claims
            and not answer.citations
        )
    elif case.case_type in {"partially_answerable", "conflicting_evidence"}:
        refusal_behavior_matches = (
            answer.status == "partially_answered"
            and bool(answer.claims)
            and bool(answer.refusal_reason)
        )
    else:
        refusal_behavior_matches = (
            answer.status == "answered"
            and bool(answer.claims)
            and not answer.refusal_reason
        )
    refusal_accuracy = float(status_matches and refusal_behavior_matches)

    dimension_values = [
        correctness,
        faithfulness,
        citation_accuracy,
        refusal_accuracy,
    ]
    if completeness is not None:
        dimension_values.append(completeness)
    answer_success = (
        all(value == 1.0 for value in dimension_values)
        and unsupported_claim_rate == 0.0
        and not answer.used_fallback
    )
    return {
        "scores": {
            "correctness": correctness,
            "faithfulness": faithfulness,
            "citation_accuracy": citation_accuracy,
            "completeness": completeness,
            "refusal_accuracy": refusal_accuracy,
            "unsupported_claim_rate": unsupported_claim_rate,
        },
        "answer_success": answer_success,
        "raw_checks": {
            "expected_claim_matches": expectation_matches,
            "generated_claim_matches": {
                str(index): matches for index, matches in claim_matches.items()
            },
            "faithfulness_errors": faithfulness_errors,
            "citation_checks": citation_checks,
            "status_matches_expectation": status_matches,
            "refusal_behavior_matches": refusal_behavior_matches,
            "displayed_claim_count": len(answer.claims),
            "unsupported_displayed_claim_count": unsupported_claim_count,
            "rejected_pre_display_claims": len(answer.rejected_claims),
        },
    }


def classify_failure(
    case: AnswerEvaluationCase,
    stage_checks: dict[str, bool | None],
    answer_success: bool,
    *,
    has_generation_evidence: bool,
) -> str | None:
    """Attribute the earliest observable failing stage for one case."""

    if answer_success:
        return None
    if case.run_mode == "synthetic_context":
        return "generation"
    if case.case_type == "unanswerable":
        return "reranking" if has_generation_evidence else "generation"
    ordered_checks = (
        ("document_ready", "extraction"),
        ("gold_evidence_indexed", "chunking"),
        ("gold_in_raw_candidates", "candidate_retrieval"),
        ("gold_in_fused_candidates", "fusion"),
        ("gold_above_relevance_cutoff", "reranking"),
        ("gold_in_generation_evidence", "evidence_selection"),
    )
    for check_name, stage in ordered_checks:
        if stage_checks.get(check_name) is False:
            return stage
    return "generation"


def _candidate_hashes(items: Iterable[Any]) -> set[str]:
    return {item.chunk_content_hash for item in items}


def _candidate_trace(candidate: Any) -> dict[str, Any]:
    return {
        "chunk_id": candidate.chunk_id,
        "chunk_content_hash": candidate.chunk_content_hash,
        "chunk_index": candidate.chunk_index,
        "page_start": candidate.page_start,
        "page_end": candidate.page_end,
        "dense_rank": getattr(candidate, "dense_rank", None),
        "lexical_rank": getattr(candidate, "lexical_rank", None),
        "fused_rank": getattr(candidate, "fused_rank", None),
        "expanded_dense_rank": getattr(candidate, "expanded_dense_rank", None),
        "expanded_lexical_rank": getattr(candidate, "expanded_lexical_rank", None),
    }


def _pipeline_stage_trace(
    case: AnswerEvaluationCase,
    result: GroundedQueryResult,
    db: Session,
) -> tuple[dict[str, Any], bool | None]:
    document = (
        db.query(Document)
        .filter(
            Document.original_filename == case.filename,
            Document.content_hash == case.document_content_hash,
        )
        .one_or_none()
    )
    indexed_hashes: set[str] = set()
    if document is not None:
        indexed_hashes = {
            content_hash
            for (content_hash,) in db.query(DocumentChunk.content_hash)
            .filter(DocumentChunk.document_id == document.id)
            .all()
        }

    original = result.retrieval.original_result
    expanded = result.retrieval.expanded_result
    raw_candidates = list(original.dense_candidates) + list(
        original.lexical_candidates
    )
    if expanded is not None:
        raw_candidates.extend(expanded.dense_candidates)
        raw_candidates.extend(expanded.lexical_candidates)
    expected = set(case.expected_evidence_hashes)
    fused = list(result.retrieval.fused_candidates)
    reranked = list(result.reranking.ranked_candidates)
    above_cutoff = [
        item.candidate
        for item in reranked
        if item.reranker_score is not None
        and item.reranker_score >= DEFAULT_RELEVANCE_CUTOFF
    ]
    generation_evidence = list(result.context.evidence)
    if case.case_type == "unanswerable":
        retrieval_success: bool | None = None
        checks: dict[str, bool | None] = {
            "document_ready": bool(
                document is not None and document.status == DOCUMENT_STATUS_READY
            ),
            "gold_evidence_indexed": None,
            "gold_in_raw_candidates": None,
            "gold_in_fused_candidates": None,
            "gold_above_relevance_cutoff": None,
            "gold_in_generation_evidence": None,
        }
    else:
        checks = {
            "document_ready": bool(
                document is not None and document.status == DOCUMENT_STATUS_READY
            ),
            "gold_evidence_indexed": bool(expected & indexed_hashes),
            "gold_in_raw_candidates": bool(expected & _candidate_hashes(raw_candidates)),
            "gold_in_fused_candidates": bool(expected & _candidate_hashes(fused)),
            "gold_above_relevance_cutoff": bool(
                expected & _candidate_hashes(above_cutoff)
            ),
            "gold_in_generation_evidence": bool(
                expected
                & {item.chunk_content_hash for item in generation_evidence}
            ),
        }
        retrieval_success = bool(checks["gold_in_fused_candidates"])
    return (
        {
            "checks": checks,
            "original_dense_candidates": [
                _candidate_trace(item) for item in original.dense_candidates
            ],
            "original_lexical_candidates": [
                _candidate_trace(item) for item in original.lexical_candidates
            ],
            "expanded_dense_candidates": (
                [_candidate_trace(item) for item in expanded.dense_candidates]
                if expanded is not None
                else []
            ),
            "expanded_lexical_candidates": (
                [_candidate_trace(item) for item in expanded.lexical_candidates]
                if expanded is not None
                else []
            ),
            "final_fused_candidates": [_candidate_trace(item) for item in fused],
            "reranked_candidates": [
                {
                    **_candidate_trace(item.candidate),
                    "input_fused_rank": item.original_fused_rank,
                    "reranked_rank": item.final_rank,
                    "reranker_score": item.reranker_score,
                    "reranker_rationale": item.reranker_rationale,
                }
                for item in reranked
            ],
        },
        retrieval_success,
    )


def _synthetic_context(case: AnswerEvaluationCase) -> GenerationContext:
    evidence = tuple(
        EvidenceItem(
            source_id=f"S{index}",
            chunk_id=f"phase11-synthetic-{case.case_id}-{index}",
            chunk_content_hash=item.content_hash,
            document_id=f"phase11-synthetic-{case.case_id}",
            document_content_hash=case.document_content_hash,
            filename=case.filename,
            chunk_index=index - 1,
            page_start=item.page_start,
            page_end=item.page_end,
            section_title=item.section_title,
            passage_text=item.passage_text,
            reranker_score=3,
            reranked_rank=index,
            original_fused_rank=index,
        )
        for index, item in enumerate(case.synthetic_evidence, start=1)
    )
    return GenerationContext(evidence=evidence, suppressed=())


def _answer_payload(answer: GroundedAnswer) -> dict[str, Any]:
    return {
        "status": answer.status,
        "answer": answer.answer,
        "claims": [asdict(item) for item in answer.claims],
        "citations": [asdict(item) for item in answer.citations],
        "refusal_reason": answer.refusal_reason,
        "used_fallback": answer.used_fallback,
        "fallback_error": answer.fallback_error,
        "rejected_claims": [asdict(item) for item in answer.rejected_claims],
        "usage": asdict(answer.usage),
        "latency_ms": round(answer.latency_ms, 3),
    }


def _evidence_payload(context: GenerationContext) -> list[dict[str, Any]]:
    """Preserve the exact passage sent to the answer model, not a preview."""

    return [asdict(item) for item in context.evidence]


def _run_case(
    case: AnswerEvaluationCase,
    db: Session,
) -> dict[str, Any]:
    started = time.perf_counter()
    if case.run_mode == "pipeline":
        result = answer_document_question(
            case.question,
            db,
            case.filename,
            document_content_hash=case.document_content_hash,
        )
        answer = result.answer
        context = result.context
        stage_trace, retrieval_success = _pipeline_stage_trace(case, result, db)
        usage = {
            "expansion": asdict(result.retrieval.expansion.usage),
            "reranker": asdict(result.reranking.usage),
            "answer": asdict(result.answer.usage),
        }
        diagnostics = {
            "expansion": asdict(result.retrieval.expansion),
            "reranker_used_fallback": result.reranking.used_fallback,
            "reranker_fallback_error": result.reranking.fallback_error,
            "generation_used_fallback": result.answer.used_fallback,
            "generation_fallback_error": result.answer.fallback_error,
        }
    else:
        context = _synthetic_context(case)
        answer = generate_grounded_answer(case.question, context)
        stage_trace = {
            "checks": {
                "document_ready": None,
                "gold_evidence_indexed": None,
                "gold_in_raw_candidates": None,
                "gold_in_fused_candidates": None,
                "gold_above_relevance_cutoff": None,
                "gold_in_generation_evidence": True,
            },
            "retrieval_bypassed_for_controlled_generation_case": True,
        }
        retrieval_success = None
        usage = {"answer": asdict(answer.usage)}
        diagnostics = {
            "generation_used_fallback": answer.used_fallback,
            "generation_fallback_error": answer.fallback_error,
        }

    grade = grade_answer(case, answer, context)
    failure_stage = classify_failure(
        case,
        stage_trace["checks"],
        grade["answer_success"],
        has_generation_evidence=bool(context.evidence),
    )
    return {
        "id": case.case_id,
        "run_mode": case.run_mode,
        "case_type": case.case_type,
        "filename": case.filename,
        "document_content_hash": case.document_content_hash,
        "question": case.question,
        "reference_answer": case.reference_answer,
        "expected_statuses": list(case.expected_statuses),
        "expected_evidence_hashes": list(case.expected_evidence_hashes),
        "expected_claims": [asdict(item) for item in case.claims],
        "tags": list(case.tags),
        "answer_output": _answer_payload(answer),
        "exact_generation_evidence": _evidence_payload(context),
        "stage_trace": stage_trace,
        "retrieval_success": retrieval_success,
        "answer_success": grade["answer_success"],
        "failure_stage": failure_stage,
        "scores": grade["scores"],
        "raw_grader_output": grade["raw_checks"],
        "diagnostics": diagnostics,
        "usage": usage,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _macro_average(
    rows: Sequence[dict[str, Any]],
    metric: str,
) -> float | None:
    values = [
        row["scores"][metric]
        for row in rows
        if row["scores"][metric] is not None
    ]
    return sum(values) / len(values) if values else None


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate_scores = {
        metric: _macro_average(rows, metric)
        for metric in (
            "correctness",
            "faithfulness",
            "citation_accuracy",
            "completeness",
            "refusal_accuracy",
            "unsupported_claim_rate",
        )
    }
    retrieval_rows = [
        row for row in rows if row["retrieval_success"] is not None
    ]
    failure_counts: dict[str, int] = {}
    for row in rows:
        stage = row["failure_stage"] or "none"
        failure_counts[stage] = failure_counts.get(stage, 0) + 1
    return {
        **aggregate_scores,
        "retrieval_success_rate": (
            sum(row["retrieval_success"] for row in retrieval_rows)
            / len(retrieval_rows)
            if retrieval_rows
            else None
        ),
        "answer_success_rate": (
            sum(row["answer_success"] for row in rows) / len(rows)
        ),
        "answer_successes": sum(row["answer_success"] for row in rows),
        "total_cases": len(rows),
        "failure_stage_counts": failure_counts,
    }


def _git_command(*arguments: str) -> str | None:
    safe_directory = str(Path.cwd().resolve()).replace("\\", "/")
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={safe_directory}", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_commit() -> str | None:
    return _git_command("rev-parse", "HEAD") or None


def _working_tree_dirty() -> bool | None:
    output = _git_command("status", "--porcelain")
    return None if output is None else bool(output)


def _implementation_hashes() -> dict[str, str]:
    paths = (
        Path("evaluation/answer_dataset.py"),
        Path("evaluation/answer_evaluation.py"),
        Path("evaluate_answers.py"),
    )
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }


def run_answer_evaluation(
    dataset: AnswerEvaluationDataset,
    db: Session,
) -> dict[str, Any]:
    """Run bounded cases and preserve inputs, outputs, labels, and diagnostics."""

    started_at = datetime.now(timezone.utc)
    rows = []
    for case in dataset.cases:
        print(f"[{case.case_id}] running answer evaluation...", flush=True)
        rows.append(_run_case(case, db))
    completed_at = datetime.now(timezone.utc)
    configuration = {
        "embedding_model": EMBEDDING_MODEL,
        "expansion_model": DEFAULT_EXPANSION_MODEL,
        "reranker_model": DEFAULT_RERANKER_MODEL,
        "answer_model": DEFAULT_ANSWER_MODEL,
        "dense_candidate_k": DEFAULT_DENSE_CANDIDATE_K,
        "lexical_candidate_k": DEFAULT_LEXICAL_CANDIDATE_K,
        "rrf_k": DEFAULT_RRF_K,
        "rerank_k": DEFAULT_RERANK_K,
        "final_evidence_k": DEFAULT_FINAL_EVIDENCE_K,
        "relevance_cutoff": DEFAULT_RELEVANCE_CUTOFF,
        "reranker_batch_size": DEFAULT_RERANKER_BATCH_SIZE,
        "grader_version": GRADER_VERSION,
        "model_assisted_judge": MODEL_JUDGE,
    }
    seed = json.dumps(
        {
            "dataset_sha256": dataset.sha256,
            "configuration": configuration,
            "cases": rows,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "run": {
            "run_id": hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12],
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_ms": round(
                (completed_at - started_at).total_seconds() * 1000, 3
            ),
            "git_commit": _git_commit(),
            "working_tree_dirty": _working_tree_dirty(),
            "implementation_sha256": _implementation_hashes(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset": {
            "name": dataset.name,
            "description": dataset.description,
            "split": dataset.split,
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "source_retrieval_dataset": dataset.source_retrieval_dataset,
            "label_policy": dataset.label_policy,
            "case_count": len(dataset.cases),
        },
        "configuration": configuration,
        "rubric": {
            "correctness": "Fraction of displayed claims matching a fixed gold fact; an unanswerable refusal scores one only when it displays no claim.",
            "faithfulness": "Fraction of displayed claims passing the Phase 10 source, quote, and claim-support validator.",
            "citation_accuracy": "Fraction of claim-source links mapping to sent evidence, passing support validation, and using allowed gold evidence when specified.",
            "completeness": "Fraction of required gold facts represented by at least one displayed claim; not applicable to unanswerable cases.",
            "refusal_accuracy": "Whether answer status and refusal/qualification behavior match the fixed case type.",
            "unsupported_claim_rate": "Displayed claims failing the grounding validator divided by displayed claims; rejected pre-display claims are reported separately.",
        },
        "aggregate": _aggregate_rows(rows),
        "cases": rows,
    }


def _answer_from_report_payload(payload: dict[str, Any]) -> GroundedAnswer:
    claims = tuple(
        GroundedClaim(
            text=item["text"],
            source_ids=tuple(item["source_ids"]),
            supporting_quotes=tuple(item["supporting_quotes"]),
        )
        for item in payload["claims"]
    )
    citations = tuple(
        Citation(
            **{
                **item,
                "supported_claims": tuple(item["supported_claims"]),
                "supporting_quotes": tuple(item["supporting_quotes"]),
            }
        )
        for item in payload["citations"]
    )
    rejected = tuple(
        GroundedClaim(
            text=item["text"],
            source_ids=tuple(item["source_ids"]),
            supporting_quotes=tuple(item["supporting_quotes"]),
        )
        for item in payload["rejected_claims"]
    )
    return GroundedAnswer(
        status=payload["status"],
        answer=payload["answer"],
        claims=claims,
        citations=citations,
        refusal_reason=payload["refusal_reason"],
        usage=GenerationUsage(**payload["usage"]),
        latency_ms=payload["latency_ms"],
        used_fallback=payload["used_fallback"],
        fallback_error=payload["fallback_error"],
        rejected_claims=rejected,
    )


def regrade_answer_report(
    report: dict[str, Any],
    dataset: AnswerEvaluationDataset,
) -> dict[str, Any]:
    """Reapply deterministic labels to preserved outputs without model calls."""

    case_by_id = {case.case_id: case for case in dataset.cases}
    report_ids = {row["id"] for row in report.get("cases", [])}
    if report_ids != set(case_by_id):
        raise ValueError("report case IDs do not match the supplied dataset")
    for row in report["cases"]:
        case = case_by_id[row["id"]]
        context = GenerationContext(
            evidence=tuple(
                EvidenceItem(**item) for item in row["exact_generation_evidence"]
            ),
            suppressed=(),
        )
        answer = _answer_from_report_payload(row["answer_output"])
        grade = grade_answer(case, answer, context)
        row["scores"] = grade["scores"]
        row["raw_grader_output"] = grade["raw_checks"]
        row["answer_success"] = grade["answer_success"]
        row["failure_stage"] = classify_failure(
            case,
            row["stage_trace"]["checks"],
            grade["answer_success"],
            has_generation_evidence=bool(context.evidence),
        )
    report["aggregate"] = _aggregate_rows(report["cases"])
    report["dataset"]["path"] = str(dataset.path)
    report["dataset"]["sha256"] = dataset.sha256
    report["configuration"]["grader_version"] = GRADER_VERSION
    report["configuration"]["model_assisted_judge"] = MODEL_JUDGE
    # The saved run metadata describes when retrieval/generation happened and
    # must remain immutable. Record the deterministic regrade separately so a
    # later grader review cannot masquerade as a new model run.
    report["regrade"] = {
        "regraded_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "working_tree_dirty": _working_tree_dirty(),
        "implementation_sha256": _implementation_hashes(),
        "grader_version": GRADER_VERSION,
    }
    return report


def render_answer_evaluation_markdown(report: dict[str, Any]) -> str:
    """Render separate aggregate dimensions and inspectable per-case traces."""

    aggregate = report["aggregate"]
    lines = [
        f"# Answer Evaluation — {report['dataset']['name']}",
        "",
        "## Run",
        "",
        f"- Run ID: `{report['run']['run_id']}`",
        f"- Dataset SHA-256: `{report['dataset']['sha256']}`",
        f"- Dataset split: `{report['dataset']['split']}`",
        f"- Git commit: `{report['run']['git_commit']}`",
        f"- Deterministic grader: `{report['configuration']['grader_version']}`",
        "- Model-assisted judge: `not used`",
        "",
        "## Separate outcomes",
        "",
        f"- Retrieval success rate (answerable live-pipeline cases only): `{aggregate['retrieval_success_rate']:.4f}`",
        f"- Answer success rate (all cases): `{aggregate['answer_success_rate']:.4f}`",
        "",
        "| Answer dimension | Score |",
        "| --- | ---: |",
    ]
    for metric in (
        "correctness",
        "faithfulness",
        "citation_accuracy",
        "completeness",
        "refusal_accuracy",
        "unsupported_claim_rate",
    ):
        value = aggregate[metric]
        rendered = "not applicable" if value is None else f"{value:.4f}"
        lines.append(f"| {metric} | {rendered} |")
    lines.extend(
        [
            "",
            "The scores use fixed gold facts/evidence and deterministic checks. No model judge decides acceptance.",
            "",
            "## Failure attribution",
            "",
        ]
    )
    for stage, count in sorted(aggregate["failure_stage_counts"].items()):
        lines.append(f"- `{stage}`: {count}")
    lines.extend(["", "## Per-case diagnostics", ""])
    for row in report["cases"]:
        scores = row["scores"]
        completeness_display = (
            "not applicable"
            if scores["completeness"] is None
            else f"{scores['completeness']:.4f}"
        )
        evidence_pages = [
            (item["page_start"], item["page_end"])
            for item in row["exact_generation_evidence"]
        ]
        lines.extend(
            [
                f"### {row['id']}",
                "",
                f"- Type / mode: `{row['case_type']}` / `{row['run_mode']}`",
                f"- Question: {row['question']}",
                f"- Reference: {row['reference_answer']}",
                f"- Answer: {row['answer_output']['answer']}",
                f"- Status: `{row['answer_output']['status']}`",
                f"- Retrieval success: `{row['retrieval_success']}`",
                f"- Answer success: `{str(row['answer_success']).lower()}`",
                f"- Failure stage: `{row['failure_stage'] or 'none'}`",
                f"- Evidence pages: `{evidence_pages}`",
                f"- Exact evidence records in JSON: `{len(row['exact_generation_evidence'])}`",
                f"- Correctness / faithfulness / citations: `{scores['correctness']:.4f}` / `{scores['faithfulness']:.4f}` / `{scores['citation_accuracy']:.4f}`",
                f"- Completeness: `{completeness_display}`",
                f"- Refusal accuracy / unsupported-claim rate: `{scores['refusal_accuracy']:.4f}` / `{scores['unsupported_claim_rate']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproducibility and interpretation",
            "",
            f"- Dataset: `{report['dataset']['path']}`",
            f"- Source retrieval labels: `{report['dataset']['source_retrieval_dataset']}`",
            f"- Configuration: `{json.dumps(report['configuration'], sort_keys=True)}`",
            "- The local Git-ignored JSON report preserves full generation evidence, raw answers, citations, raw deterministic grader checks, and stage traces.",
            "- Synthetic cases bypass retrieval intentionally; they isolate partial/conflicting generation and therefore have no retrieval-success score.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_answer_reports(
    report: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write the machine-readable record and matching human-readable summary."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_answer_evaluation_markdown(report),
        encoding="utf-8",
    )
