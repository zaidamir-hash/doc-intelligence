from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from dense_retrieval import DenseCandidate
from expanded_retrieval import fuse_query_results, retrieve_expanded_hybrid_candidates
from hybrid_retrieval import FusedCandidate, HybridRetrievalResult
from lexical_retrieval import LexicalCandidate
from query_expansion import (
    ExpansionUsage,
    GeneratedExpansion,
    expand_query,
    extract_protected_terms,
    validate_expansion,
)


def dense(index: int, rank: int) -> DenseCandidate:
    return DenseCandidate(
        chunk_id=f"chunk-{index}",
        chunk_content_hash=f"{index:064x}",
        document_id="document-1",
        document_content_hash="d" * 64,
        chunk_index=index,
        filename="document.pdf",
        content=f"content {index}",
        passage_text=f"passage {index}",
        page_start=1,
        page_end=1,
        section_title="Section",
        dense_rank=rank,
        distance_metric="cosine",
        distance=0.2,
        similarity=0.8,
    )


def lexical(index: int, rank: int) -> LexicalCandidate:
    return LexicalCandidate(
        chunk_id=f"chunk-{index}",
        chunk_content_hash=f"{index:064x}",
        document_id="document-1",
        document_content_hash="d" * 64,
        chunk_index=index,
        filename="document.pdf",
        content=f"content {index}",
        passage_text=f"passage {index}",
        page_start=1,
        page_end=1,
        section_title="Section",
        lexical_rank=rank,
        lexical_score=0.7,
        fts_score=0.4,
        exact_match_count=1,
        exact_terms_matched=("TERM",),
        parsed_search_text="TERM",
    )


def hybrid(
    dense_candidates: list[DenseCandidate],
    lexical_candidates: list[LexicalCandidate],
) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        dense_candidates=tuple(dense_candidates),
        lexical_candidates=tuple(lexical_candidates),
        fused_candidates=(),
    )


def fused(index: int) -> FusedCandidate:
    item = dense(index, 1)
    return FusedCandidate(
        chunk_id=item.chunk_id,
        chunk_content_hash=item.chunk_content_hash,
        document_id=item.document_id,
        document_content_hash=item.document_content_hash,
        chunk_index=item.chunk_index,
        filename=item.filename,
        content=item.content,
        passage_text=item.passage_text,
        page_start=item.page_start,
        page_end=item.page_end,
        section_title=item.section_title,
        fused_rank=1,
        fused_score=0.1,
        dense_rank=1,
        dense_distance=0.2,
        dense_similarity=0.8,
        dense_rrf_contribution=0.05,
        lexical_rank=None,
        lexical_score=None,
        fts_score=None,
        exact_match_count=0,
        exact_terms_matched=(),
        parsed_search_text=None,
        lexical_rrf_contribution=0.0,
        source_count=1,
    )


class ExpansionValidationTests(unittest.TestCase):
    def test_extracts_names_numbers_dates_acronyms_and_identifiers(self) -> None:
        terms = extract_protected_terms(
            "What were Pakistan's NCPI rates in FY25 and FY24 for Climate Risk Fund-I?"
        )

        folded = {term.casefold() for term in terms}
        self.assertTrue(any("pakistan's" in term for term in folded))
        self.assertIn("ncpi", folded)
        self.assertIn("fy25", folded)
        self.assertIn("fy24", folded)
        self.assertIn("climate risk fund-i", folded)

    def test_valid_expansion_preserves_intent_and_exact_signals(self) -> None:
        original = "What does the acronym PRISM+ stand for?"
        generated = GeneratedExpansion(
            alternative_query="Full form and expansion of the acronym PRISM+",
            lexical_terms=("full form", "PRISM+"),
            intent_preserved=True,
            rationale="Uses likely document wording.",
        )

        self.assertEqual(
            validate_expansion(original, generated),
            "Full form and expansion of the acronym PRISM+",
        )

    def test_missing_protected_term_is_rejected(self) -> None:
        generated = GeneratedExpansion(
            alternative_query="What was the inflation rate?",
            lexical_terms=(),
            intent_preserved=True,
            rationale="",
        )

        with self.assertRaisesRegex(ValueError, "removed protected terms"):
            validate_expansion("What was NCPI inflation in FY25?", generated)

    def test_new_number_or_acronym_is_rejected(self) -> None:
        generated = GeneratedExpansion(
            alternative_query="How did public debt change in FY25 and FY26?",
            lexical_terms=(),
            intent_preserved=True,
            rationale="",
        )

        with self.assertRaisesRegex(ValueError, "introduced new exact signals"):
            validate_expansion("How did public debt change in FY25?", generated)

    def test_empty_identical_and_drifted_expansions_fall_back(self) -> None:
        for alternative, intent in (("Question about banking", False), ("", True)):
            result = expand_query(
                "What is banking regulation?",
                generator=lambda _query, _terms, _model: GeneratedExpansion(
                    alternative_query=alternative,
                    lexical_terms=(),
                    intent_preserved=intent,
                    rationale="",
                ),
            )
            self.assertFalse(result.used_expansion)
            self.assertEqual(result.original_query, "What is banking regulation?")
            self.assertEqual(result.generated_query, alternative)
            self.assertIsNotNone(result.fallback_reason)

    def test_rejected_generation_remains_visible_and_counted(self) -> None:
        result = expand_query(
            "What was NCPI inflation in FY25?",
            generator=lambda _query, _terms, _model: GeneratedExpansion(
                alternative_query="What was inflation in FY25?",
                lexical_terms=("inflation",),
                intent_preserved=True,
                rationale="Attempted a shorter query.",
                usage=ExpansionUsage(1, 20, 8, 0.0000078),
            ),
        )

        self.assertFalse(result.used_expansion)
        self.assertEqual(result.generated_query, "What was inflation in FY25?")
        self.assertIsNone(result.expanded_query)
        self.assertEqual(result.lexical_terms, ("inflation",))
        self.assertEqual(result.usage.requests, 1)

    def test_valid_generator_output_is_visible(self) -> None:
        result = expand_query(
            "How did monetary easing affect policy rates?",
            generator=lambda _query, _terms, _model: GeneratedExpansion(
                alternative_query="Policy rate reductions caused by monetary easing",
                lexical_terms=("policy rate", "monetary easing"),
                intent_preserved=True,
                rationale="Adds likely report wording.",
                usage=ExpansionUsage(1, 20, 10, 0.000009),
            ),
        )

        self.assertTrue(result.used_expansion)
        self.assertEqual(result.lexical_terms, ("policy rate", "monetary easing"))
        self.assertEqual(result.usage.requests, 1)


class MultiQueryFusionTests(unittest.TestCase):
    def test_four_signals_are_fused_once_per_stable_chunk(self) -> None:
        original = hybrid([dense(1, 1), dense(2, 2)], [lexical(1, 2)])
        alternative = hybrid([dense(1, 3), dense(3, 1)], [lexical(1, 1)])

        result = fuse_query_results(original, alternative, rrf_k=10)

        self.assertEqual(len(result), 3)
        shared = next(item for item in result if item.chunk_index == 1)
        self.assertEqual(shared.source_count, 4)
        self.assertAlmostEqual(
            shared.fused_score, 1 / 11 + 1 / 12 + 1 / 13 + 1 / 11
        )
        self.assertEqual(shared.dense_rank, 1)
        self.assertEqual(shared.expanded_dense_rank, 3)
        self.assertEqual(shared.expanded_lexical_rank, 1)

    @patch("expanded_retrieval.retrieve_hybrid_candidates")
    def test_failed_expansion_returns_exact_original_pipeline(self, retrieve) -> None:
        original_candidate = fused(1)
        retrieve.return_value = HybridRetrievalResult(
            dense_candidates=(),
            lexical_candidates=(),
            fused_candidates=(original_candidate,),
        )
        generator = Mock(side_effect=RuntimeError("generator offline"))

        result = retrieve_expanded_hybrid_candidates(
            "What is the policy rate?",
            Mock(),
            "document.pdf",
            expansion_generator=generator,
        )

        self.assertEqual(retrieve.call_count, 1)
        self.assertFalse(result.expansion.used_expansion)
        self.assertEqual(result.fused_candidates, (original_candidate,))
        self.assertIn("generator offline", result.expansion.fallback_reason)

    @patch("expanded_retrieval.retrieve_hybrid_candidates")
    def test_original_is_searched_before_valid_expansion(self, retrieve) -> None:
        retrieve.side_effect = [hybrid([dense(1, 1)], []), hybrid([dense(2, 1)], [])]

        result = retrieve_expanded_hybrid_candidates(
            "How did monetary easing affect policy rates?",
            Mock(),
            "document.pdf",
            expansion_generator=lambda _query, _terms, _model: GeneratedExpansion(
                alternative_query="Policy rate reductions caused by monetary easing",
                lexical_terms=("policy rate",),
                intent_preserved=True,
                rationale="",
            ),
        )

        self.assertEqual(retrieve.call_args_list[0].args[0], "How did monetary easing affect policy rates?")
        self.assertEqual(
            retrieve.call_args_list[1].args[0],
            "Policy rate reductions caused by monetary easing",
        )
        self.assertTrue(result.expansion.used_expansion)


if __name__ == "__main__":
    unittest.main()
