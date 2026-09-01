"""Focused Phase 11 dataset, metric, and failure-attribution tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.answer_dataset import (
    AnswerEvaluationCase,
    ClaimExpectation,
    load_answer_dataset,
)
from evaluation.answer_evaluation import (
    claim_matches_expectation,
    classify_failure,
    grade_answer,
)
from grounded_generation import (
    Citation,
    EvidenceItem,
    GenerationContext,
    GenerationUsage,
    GroundedAnswer,
    GroundedClaim,
)


HASH_1 = "1" * 64
HASH_2 = "2" * 64
DOCUMENT_HASH = "d" * 64


def _expectation(
    claim_id: str,
    value: str,
    content_hash: str,
) -> ClaimExpectation:
    return ClaimExpectation(
        claim_id=claim_id,
        description=f"The reported value was {value} percent.",
        required=True,
        term_groups=(("reported value",), (value,), ("percent",)),
        evidence_hashes=(content_hash,),
    )


def _case(
    *claims: ClaimExpectation,
    case_type: str = "answerable",
    run_mode: str = "synthetic_context",
) -> AnswerEvaluationCase:
    return AnswerEvaluationCase(
        case_id="test-case",
        run_mode=run_mode,  # type: ignore[arg-type]
        case_type=case_type,
        filename="test.txt",
        question="What values were reported?",
        document_content_hash=DOCUMENT_HASH,
        reference_answer=None if case_type == "unanswerable" else "Reference",
        expected_statuses=(
            ("insufficient_evidence",)
            if case_type == "unanswerable"
            else (
                ("partially_answered",)
                if case_type in {"partially_answerable", "conflicting_evidence"}
                else ("answered",)
            )
        ),
        expected_evidence_hashes=tuple(
            dict.fromkeys(
                content_hash
                for claim in claims
                for content_hash in claim.evidence_hashes
            )
        ),
        claims=claims,
        synthetic_evidence=(),
        tags=(),
        notes=None,
    )


def _evidence(source_id: str, content_hash: str, value: str) -> EvidenceItem:
    return EvidenceItem(
        source_id=source_id,
        chunk_id=f"chunk-{source_id}",
        chunk_content_hash=content_hash,
        document_id="document-1",
        document_content_hash=DOCUMENT_HASH,
        filename="test.txt",
        chunk_index=int(source_id[1:]),
        page_start=int(source_id[1:]),
        page_end=int(source_id[1:]),
        section_title="Results",
        passage_text=f"The reported value was {value} percent.",
        reranker_score=3,
        reranked_rank=int(source_id[1:]),
        original_fused_rank=int(source_id[1:]),
    )


def _claim(source_id: str, value: str) -> GroundedClaim:
    text = f"The reported value was {value} percent."
    return GroundedClaim(
        text=text,
        source_ids=(source_id,),
        supporting_quotes=(text,),
    )


def _citation(
    evidence: EvidenceItem,
    claim: GroundedClaim,
) -> Citation:
    return Citation(
        source_id=evidence.source_id,
        chunk_id=evidence.chunk_id,
        chunk_content_hash=evidence.chunk_content_hash,
        document_id=evidence.document_id,
        document_content_hash=evidence.document_content_hash,
        filename=evidence.filename,
        chunk_index=evidence.chunk_index,
        page_start=evidence.page_start,
        page_end=evidence.page_end,
        section_title=evidence.section_title,
        preview=evidence.passage_text,
        supported_claims=(claim.text,),
        supporting_quotes=claim.supporting_quotes,
    )


def _answer(
    claims: tuple[GroundedClaim, ...],
    citations: tuple[Citation, ...],
    *,
    status: str = "answered",
    refusal_reason: str | None = None,
) -> GroundedAnswer:
    return GroundedAnswer(
        status=status,  # type: ignore[arg-type]
        answer=" ".join(claim.text for claim in claims) or "Cannot answer.",
        claims=claims,
        citations=citations,
        refusal_reason=refusal_reason,
        usage=GenerationUsage(),
        latency_ms=1.0,
    )


class AnswerDatasetTests(unittest.TestCase):
    def test_phase11_dataset_is_versioned_and_covers_required_case_types(self):
        dataset = load_answer_dataset(
            Path("evaluation/datasets/phase11_answers.json")
        )

        self.assertEqual(dataset.schema_version, 1)
        self.assertEqual(dataset.split, "test")
        self.assertEqual(len(dataset.sha256), 64)
        self.assertEqual(
            {case.case_type for case in dataset.cases},
            {
                "answerable",
                "unanswerable",
                "partially_answerable",
                "conflicting_evidence",
            },
        )
        self.assertTrue(
            all(case.expected_statuses for case in dataset.cases)
        )


class AnswerMetricTests(unittest.TestCase):
    def test_term_groups_are_explicit_and_tolerate_listed_variants(self):
        expectation = ClaimExpectation(
            claim_id="provision",
            description="Statutory provision",
            required=True,
            term_groups=(
                ("Section 39(1)", "Section 39 1"),
                ("1956",),
            ),
            evidence_hashes=(HASH_1,),
        )

        self.assertTrue(
            claim_matches_expectation(
                "The report cites Section 39(1) of the 1956 Act.",
                expectation,
            )
        )
        self.assertFalse(
            claim_matches_expectation("The report cites Section 40.", expectation)
        )

    def test_numeric_gold_terms_match_whole_tokens_not_substrings(self):
        expectation = _expectation("value_10", "10", HASH_1)

        self.assertTrue(
            claim_matches_expectation(
                "The reported value was 10 percent.", expectation
            )
        )
        self.assertFalse(
            claim_matches_expectation(
                "The reported value was 110 percent.", expectation
            )
        )

    def test_correct_but_incomplete_answer_keeps_dimensions_separate(self):
        expected_10 = _expectation("value_10", "10", HASH_1)
        expected_12 = _expectation("value_12", "12", HASH_2)
        case = _case(expected_10, expected_12)
        evidence = _evidence("S1", HASH_1, "10")
        claim = _claim("S1", "10")
        answer = _answer((claim,), (_citation(evidence, claim),))

        grade = grade_answer(
            case,
            answer,
            GenerationContext(evidence=(evidence,), suppressed=()),
        )

        self.assertEqual(grade["scores"]["correctness"], 1.0)
        self.assertEqual(grade["scores"]["completeness"], 0.5)
        self.assertEqual(grade["scores"]["faithfulness"], 1.0)
        self.assertEqual(grade["scores"]["citation_accuracy"], 1.0)
        self.assertFalse(grade["answer_success"])

    def test_wrong_gold_evidence_is_distinct_from_faithfulness(self):
        expectation = _expectation("value_10", "10", HASH_1)
        case = _case(expectation)
        different_evidence = _evidence("S1", HASH_2, "10")
        claim = _claim("S1", "10")
        answer = _answer(
            (claim,),
            (_citation(different_evidence, claim),),
        )

        grade = grade_answer(
            case,
            answer,
            GenerationContext(evidence=(different_evidence,), suppressed=()),
        )

        self.assertEqual(grade["scores"]["faithfulness"], 1.0)
        self.assertEqual(grade["scores"]["citation_accuracy"], 0.0)
        self.assertEqual(grade["scores"]["correctness"], 1.0)

    def test_unsupported_displayed_claim_is_measured(self):
        expectation = _expectation("value_10", "10", HASH_1)
        case = _case(expectation)
        evidence = _evidence("S1", HASH_1, "10")
        unsupported = GroundedClaim(
            text="A Bitcoin reserve target was announced.",
            source_ids=("S1",),
            supporting_quotes=("The reported value was 10 percent.",),
        )
        answer = _answer(
            (unsupported,),
            (_citation(evidence, unsupported),),
        )

        grade = grade_answer(
            case,
            answer,
            GenerationContext(evidence=(evidence,), suppressed=()),
        )

        self.assertEqual(grade["scores"]["correctness"], 0.0)
        self.assertEqual(grade["scores"]["faithfulness"], 0.0)
        self.assertEqual(grade["scores"]["unsupported_claim_rate"], 1.0)

    def test_unanswerable_refusal_scores_without_fabricated_completeness(self):
        case = _case(case_type="unanswerable")
        answer = _answer(
            (),
            (),
            status="insufficient_evidence",
            refusal_reason="No evidence met the cutoff.",
        )

        grade = grade_answer(
            case,
            answer,
            GenerationContext(evidence=(), suppressed=()),
        )

        self.assertEqual(grade["scores"]["refusal_accuracy"], 1.0)
        self.assertIsNone(grade["scores"]["completeness"])
        self.assertTrue(grade["answer_success"])


class FailureAttributionTests(unittest.TestCase):
    def test_each_pipeline_stage_can_be_attributed(self):
        case = _case(_expectation("value_10", "10", HASH_1), run_mode="pipeline")
        ordered = [
            ("document_ready", "extraction"),
            ("gold_evidence_indexed", "chunking"),
            ("gold_in_raw_candidates", "candidate_retrieval"),
            ("gold_in_fused_candidates", "fusion"),
            ("gold_above_relevance_cutoff", "reranking"),
            ("gold_in_generation_evidence", "evidence_selection"),
        ]
        for failing_check, expected_stage in ordered:
            checks = {name: True for name, _ in ordered}
            checks[failing_check] = False
            with self.subTest(failing_check=failing_check):
                self.assertEqual(
                    classify_failure(
                        case,
                        checks,
                        False,
                        has_generation_evidence=False,
                    ),
                    expected_stage,
                )

    def test_generation_is_last_stage_after_successful_retrieval(self):
        case = _case(_expectation("value_10", "10", HASH_1), run_mode="pipeline")
        checks = {
            "document_ready": True,
            "gold_evidence_indexed": True,
            "gold_in_raw_candidates": True,
            "gold_in_fused_candidates": True,
            "gold_above_relevance_cutoff": True,
            "gold_in_generation_evidence": True,
        }

        self.assertEqual(
            classify_failure(
                case,
                checks,
                False,
                has_generation_evidence=True,
            ),
            "generation",
        )
        self.assertIsNone(
            classify_failure(
                case,
                checks,
                True,
                has_generation_evidence=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
