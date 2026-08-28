from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from dense_retrieval import (
    DenseCandidate,
    DenseSelection,
    expand_adjacent_context,
    select_dense_context,
    similarity_from_distance,
    token_jaccard,
    validate_retrieval_sizes,
)


def candidate(
    index: int,
    text: str,
    *,
    similarity: float = 0.8,
    rank: int | None = None,
) -> DenseCandidate:
    return DenseCandidate(
        chunk_id=f"chunk-{index}",
        chunk_content_hash=str(index) * 64,
        document_id="00000000-0000-0000-0000-000000000001",
        document_content_hash="a" * 64,
        chunk_index=index,
        filename="document.pdf",
        content=text,
        passage_text=text,
        page_start=1,
        page_end=1,
        section_title="Section",
        dense_rank=rank or index + 1,
        distance_metric="cosine",
        distance=1.0 - similarity,
        similarity=similarity,
    )


class DenseSelectionTests(unittest.TestCase):
    def test_raw_distance_is_converted_to_interpretable_similarity(self) -> None:
        self.assertAlmostEqual(similarity_from_distance(0.25, "cosine"), 0.75)
        self.assertAlmostEqual(similarity_from_distance(0.25, "l2"), 0.8)

    def test_candidate_and_final_sizes_are_independent_and_validated(self) -> None:
        validate_retrieval_sizes(candidate_k=20, top_k=5)
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            validate_retrieval_sizes(candidate_k=4, top_k=5)

    def test_overlap_aware_suppression_records_which_chunk_won(self) -> None:
        first = candidate(0, "policy rate decreased by eleven hundred basis points")
        duplicate = candidate(
            1,
            "policy rate decreased by eleven hundred basis points during the year",
        )
        different = candidate(2, "financial inclusion reached a new target")

        selection = select_dense_context(
            [first, duplicate, different],
            top_k=2,
            duplicate_similarity_threshold=0.7,
        )

        self.assertEqual([item.chunk_id for item in selection.selected], ["chunk-0", "chunk-2"])
        self.assertEqual(selection.suppressed[0].reason, "near_duplicate")
        self.assertEqual(selection.suppressed[0].duplicate_of_chunk_id, "chunk-0")
        self.assertGreaterEqual(selection.suppressed[0].text_similarity or 0, 0.7)

    def test_relevance_threshold_can_return_no_context(self) -> None:
        selection = select_dense_context(
            [candidate(0, "weak result", similarity=0.31)],
            top_k=1,
            relevance_threshold=0.5,
        )

        self.assertEqual(selection.selected, ())
        self.assertEqual(selection.suppressed[0].reason, "below_threshold")

    def test_token_jaccard_is_case_insensitive_and_bounded(self) -> None:
        self.assertEqual(token_jaccard("Same words", "same WORDS"), 1.0)
        self.assertEqual(token_jaccard("first", "second"), 0.0)


class AdjacentExpansionTests(unittest.TestCase):
    def test_expansion_labels_neighbors_and_deduplicates_shared_neighbors(self) -> None:
        anchor = candidate(5, "anchor", rank=1)
        rows = [
            SimpleNamespace(
                id="neighbor-4",
                content_hash="4" * 64,
                document_id=anchor.document_id,
                chunk_index=4,
                content="before",
                page_start=1,
                page_end=1,
                section_title="Section",
            ),
            SimpleNamespace(
                id=anchor.chunk_id,
                content_hash=anchor.chunk_content_hash,
                document_id=anchor.document_id,
                chunk_index=5,
                content="anchor",
                page_start=1,
                page_end=1,
                section_title="Section",
            ),
            SimpleNamespace(
                id="neighbor-6",
                content_hash="6" * 64,
                document_id=anchor.document_id,
                chunk_index=6,
                content="after",
                page_start=2,
                page_end=2,
                section_title="Section",
            ),
        ]
        database = MagicMock()
        database.query.return_value.filter.return_value.order_by.return_value.all.return_value = rows

        contexts = expand_adjacent_context(
            DenseSelection((anchor,), ()), database, adjacent_window=1
        )

        self.assertEqual([item.chunk_index for item in contexts], [4, 5, 6])
        self.assertEqual([item.source for item in contexts], ["adjacent", "dense", "adjacent"])


if __name__ == "__main__":
    unittest.main()
