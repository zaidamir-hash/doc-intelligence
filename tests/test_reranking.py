from __future__ import annotations

import unittest

from hybrid_retrieval import FusedCandidate
from reranking import (
    PairScore,
    RerankerUsage,
    ScoreBatchResult,
    build_question_passage_pairs,
    estimate_cost_usd,
    rerank_candidates,
)


def candidate(index: int, fused_rank: int) -> FusedCandidate:
    return FusedCandidate(
        chunk_id=f"chunk-{index}",
        chunk_content_hash=f"{index:064x}",
        document_id="document-1",
        document_content_hash="d" * 64,
        chunk_index=index,
        filename="document.pdf",
        content=f"Page 1\n\nSection\n\npassage {index}",
        passage_text=f"passage {index}",
        page_start=1,
        page_end=1,
        section_title="Section",
        fused_rank=fused_rank,
        fused_score=0.1 / fused_rank,
        dense_rank=fused_rank,
        dense_distance=0.2,
        dense_similarity=0.8,
        dense_rrf_contribution=0.05,
        lexical_rank=fused_rank,
        lexical_score=0.7,
        fts_score=0.4,
        exact_match_count=0,
        exact_terms_matched=(),
        parsed_search_text="question",
        lexical_rrf_contribution=0.05,
        source_count=2,
    )


class RerankingTests(unittest.TestCase):
    def test_pairs_make_question_and_passage_explicit(self) -> None:
        pairs = build_question_passage_pairs("  What happened?  ", [candidate(1, 1)])

        self.assertEqual(pairs[0].question, "What happened?")
        self.assertEqual(pairs[0].pair_id, f"{1:064x}")
        self.assertEqual(pairs[0].passage, "Section: Section\n\npassage 1")

    def test_scores_reorder_candidates_and_preserve_input_rank(self) -> None:
        def scorer(pairs, _model):
            scores = {f"{1:064x}": 1, f"{2:064x}": 3, f"{3:064x}": 2}
            return ScoreBatchResult(
                scores=tuple(
                    PairScore(pair.pair_id, scores[pair.pair_id], "reason")
                    for pair in pairs
                ),
                usage=RerankerUsage(1, 100, 20, estimate_cost_usd(100, 20)),
            )

        result = rerank_candidates(
            "question",
            [candidate(1, 1), candidate(2, 2), candidate(3, 3)],
            rerank_k=3,
            final_k=2,
            relevance_cutoff=2,
            score_pairs=scorer,
        )

        self.assertEqual(
            [item.candidate.chunk_index for item in result.ranked_candidates],
            [2, 3, 1],
        )
        self.assertEqual(result.ranked_candidates[0].original_fused_rank, 2)
        self.assertEqual(result.ranked_candidates[0].final_rank, 1)
        self.assertEqual(
            [item.candidate.chunk_index for item in result.selected_candidates],
            [2, 3],
        )
        self.assertFalse(result.used_fallback)

    def test_tied_scores_keep_fused_order(self) -> None:
        def scorer(pairs, _model):
            return ScoreBatchResult(
                scores=tuple(PairScore(pair.pair_id, 2, "tie") for pair in pairs),
                usage=RerankerUsage(),
            )

        result = rerank_candidates(
            "question",
            [candidate(1, 1), candidate(2, 2)],
            rerank_k=2,
            final_k=2,
            score_pairs=scorer,
        )

        self.assertEqual(
            [item.original_fused_rank for item in result.ranked_candidates], [1, 2]
        )

    def test_batches_are_independent_and_usage_is_added(self) -> None:
        calls = []

        def scorer(pairs, _model):
            calls.append(len(pairs))
            return ScoreBatchResult(
                scores=tuple(PairScore(pair.pair_id, 2, "ok") for pair in pairs),
                usage=RerankerUsage(1, 10, 2, estimate_cost_usd(10, 2)),
            )

        result = rerank_candidates(
            "question",
            [candidate(index, index) for index in range(1, 6)],
            rerank_k=5,
            final_k=3,
            batch_size=2,
            score_pairs=scorer,
        )

        self.assertEqual(calls, [2, 2, 1])
        self.assertEqual(result.usage.requests, 3)
        self.assertEqual(result.usage.input_tokens, 30)
        self.assertEqual(result.usage.output_tokens, 6)

    def test_failure_explicitly_preserves_fused_order(self) -> None:
        def failing_scorer(_pairs, _model):
            raise RuntimeError("service unavailable")

        result = rerank_candidates(
            "question",
            [candidate(1, 1), candidate(2, 2)],
            rerank_k=2,
            final_k=1,
            relevance_cutoff=3,
            score_pairs=failing_scorer,
        )

        self.assertTrue(result.used_fallback)
        self.assertIn("service unavailable", result.fallback_error)
        self.assertEqual(result.ranked_candidates[0].original_fused_rank, 1)
        self.assertIsNone(result.ranked_candidates[0].reranker_score)
        self.assertEqual(result.selected_candidates[0].candidate.chunk_index, 1)

    def test_duplicate_or_missing_scores_trigger_visible_fallback(self) -> None:
        def invalid_scorer(pairs, _model):
            return ScoreBatchResult(
                scores=(PairScore(pairs[0].pair_id, 3, "only one"),),
                usage=RerankerUsage(),
            )

        result = rerank_candidates(
            "question",
            [candidate(1, 1), candidate(2, 2)],
            rerank_k=2,
            final_k=2,
            score_pairs=invalid_scorer,
        )

        self.assertTrue(result.used_fallback)
        self.assertIn("pair IDs", result.fallback_error)

    def test_configuration_and_cost_validation(self) -> None:
        self.assertAlmostEqual(estimate_cost_usd(1_000_000, 1_000_000), 0.75)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            estimate_cost_usd(-1, 0)
        with self.assertRaisesRegex(ValueError, "final_k cannot exceed"):
            rerank_candidates("q", [], rerank_k=1, final_k=2)
        with self.assertRaisesRegex(ValueError, "relevance_cutoff"):
            rerank_candidates("q", [], relevance_cutoff=4)


if __name__ == "__main__":
    unittest.main()
