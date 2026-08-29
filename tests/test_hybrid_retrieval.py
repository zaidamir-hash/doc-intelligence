from __future__ import annotations

import os
import unittest

from database import SessionLocal
from dense_retrieval import DenseCandidate
from hybrid_retrieval import (
    fuse_candidates,
    retrieve_hybrid_candidates,
    rrf_contribution,
)
from lexical_retrieval import LexicalCandidate


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
        fts_score=0.45,
        exact_match_count=1,
        exact_terms_matched=("TERM",),
        parsed_search_text="TERM",
    )


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_formula_matches_hand_calculation(self) -> None:
        self.assertAlmostEqual(rrf_contribution(1, 60), 1 / 61)
        self.assertAlmostEqual(rrf_contribution(3, 60), 1 / 63)
        self.assertEqual(rrf_contribution(None, 60), 0.0)

        result = fuse_candidates([dense(7, 1)], [lexical(7, 3)], rrf_k=60)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].fused_score, 1 / 61 + 1 / 63)
        self.assertAlmostEqual(result[0].dense_rrf_contribution, 1 / 61)
        self.assertAlmostEqual(result[0].lexical_rrf_contribution, 1 / 63)

    def test_stable_identity_deduplicates_shared_chunk(self) -> None:
        result = fuse_candidates(
            [dense(1, 1), dense(2, 2)],
            [lexical(1, 2), lexical(3, 1)],
            rrf_k=60,
        )

        self.assertEqual(len(result), 3)
        shared = next(candidate for candidate in result if candidate.chunk_index == 1)
        self.assertEqual(shared.source_count, 2)
        self.assertEqual(shared.dense_rank, 1)
        self.assertEqual(shared.lexical_rank, 2)

    def test_chunk_present_in_one_list_gets_zero_missing_contribution(self) -> None:
        result = fuse_candidates([dense(1, 1)], [lexical(2, 1)], rrf_k=10)

        dense_only = next(candidate for candidate in result if candidate.chunk_index == 1)
        lexical_only = next(candidate for candidate in result if candidate.chunk_index == 2)
        self.assertIsNone(dense_only.lexical_rank)
        self.assertEqual(dense_only.lexical_rrf_contribution, 0.0)
        self.assertIsNone(lexical_only.dense_rank)
        self.assertEqual(lexical_only.dense_rrf_contribution, 0.0)

    def test_raw_source_scores_and_final_rank_remain_visible(self) -> None:
        result = fuse_candidates([dense(4, 2)], [lexical(4, 1)], rrf_k=30)

        self.assertEqual(result[0].fused_rank, 1)
        self.assertEqual(result[0].dense_distance, 0.2)
        self.assertEqual(result[0].dense_similarity, 0.8)
        self.assertEqual(result[0].lexical_score, 0.7)
        self.assertEqual(result[0].fts_score, 0.45)
        self.assertEqual(result[0].exact_terms_matched, ("TERM",))

    def test_top_k_is_applied_after_union_and_fusion(self) -> None:
        result = fuse_candidates(
            [dense(1, 1), dense(2, 2)],
            [lexical(2, 1), lexical(3, 2)],
            rrf_k=60,
            top_k=1,
        )

        self.assertEqual([candidate.chunk_index for candidate in result], [2])

    def test_invalid_rrf_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rrf_k must be positive"):
            rrf_contribution(1, 0)
        with self.assertRaisesRegex(ValueError, "rank must be positive"):
            rrf_contribution(0, 60)
        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            fuse_candidates([], [], top_k=0)


@unittest.skipUnless(
    os.getenv("RUN_DATABASE_TESTS") == "1",
    "set RUN_DATABASE_TESTS=1 to run PostgreSQL integration tests",
)
class HybridPostgresTests(unittest.TestCase):
    def test_independent_candidate_lists_are_fused_with_traceable_scores(self) -> None:
        database = SessionLocal()
        try:
            result = retrieve_hybrid_candidates(
                "What does PRISM+ stand for?",
                database,
                "Gov-AR_1 test.pdf",
                dense_candidate_k=5,
                lexical_candidate_k=5,
                fused_top_k=5,
                rrf_k=10,
                embedding_function=lambda _text: [1.0] + [0.0] * 1535,
            )
        finally:
            database.close()

        self.assertEqual(len(result.dense_candidates), 5)
        self.assertEqual(len(result.lexical_candidates), 5)
        self.assertEqual(len(result.fused_candidates), 5)
        identities = [
            (candidate.document_id, candidate.chunk_content_hash)
            for candidate in result.fused_candidates
        ]
        self.assertEqual(len(identities), len(set(identities)))
        for candidate in result.fused_candidates:
            self.assertAlmostEqual(
                candidate.fused_score,
                candidate.dense_rrf_contribution
                + candidate.lexical_rrf_contribution,
            )


if __name__ == "__main__":
    unittest.main()
