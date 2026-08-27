from __future__ import annotations

import unittest

from chunking import (
    CHUNKER_VERSION,
    ChunkingConfig,
    Tokenizer,
    chunk_extraction,
    extract_structural_units,
)
from pdf_processing import process_extracted_pages
from evaluate_chunking import matching_chunks


class TokenAndStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = Tokenizer()
        self.config = ChunkingConfig(
            max_tokens=180,
            overlap_tokens=25,
            min_chunk_tokens=25,
        )

    def test_tokenizer_uses_documented_encoding(self) -> None:
        self.assertEqual(self.tokenizer.encoding_name, "cl100k_base")
        text = "Tokens are model input pieces, not words or characters."
        self.assertEqual(
            self.tokenizer.decode(self.tokenizer.encode(text)),
            text,
        )

    def test_heading_context_and_page_range_are_retained(self) -> None:
        extraction = process_extracted_pages(
            [
                "2.3 Monetary Policy\n\n"
                "The committee maintained a cautious policy stance.",
                "The following page continues the same policy discussion.",
            ]
        )

        chunks = chunk_extraction(extraction, self.config, self.tokenizer)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_title, "2.3 Monetary Policy")
        self.assertEqual((chunks[0].page_start, chunks[0].page_end), (1, 2))
        self.assertIn("[Source pages 1-2]", chunks[0].content)
        self.assertIn("[Section: 2.3 Monetary Policy]", chunks[0].content)

    def test_sections_paragraphs_and_lists_remain_visible(self) -> None:
        extraction = process_extracted_pages(
            [
                "1 Overview\n\nFirst paragraph has useful evidence.\n\n"
                "\uf0b7 First action\n\uf0b7 Second action\n\n"
                "2 Results\n\nA different section begins here."
            ]
        )
        units = extract_structural_units(extraction.pages, self.tokenizer)

        self.assertEqual(units[0].section_title, "1 Overview")
        self.assertEqual(units[0].boundary_kind, "paragraph")
        self.assertEqual(units[1].boundary_kind, "list-item")
        self.assertEqual(units[2].boundary_kind, "list-item")
        self.assertEqual(units[-1].section_title, "2 Results")

    def test_long_text_uses_safe_hard_token_boundaries_and_makes_progress(self) -> None:
        extraction = process_extracted_pages(["continuousword " * 900])

        chunks = chunk_extraction(extraction, self.config, self.tokenizer)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.passage_text.strip() for chunk in chunks))
        self.assertTrue(
            all(chunk.token_count <= self.config.max_tokens for chunk in chunks)
        )
        self.assertTrue(
            any("hard-token-boundary" in chunk.boundary_kinds for chunk in chunks)
        )

    def test_overlap_is_bounded_and_recorded(self) -> None:
        paragraphs = "\n\n".join(
            f"Paragraph {number} explains a separate but related policy detail."
            for number in range(30)
        )
        extraction = process_extracted_pages([paragraphs])

        chunks = chunk_extraction(extraction, self.config, self.tokenizer)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(any(chunk.overlap_token_count > 0 for chunk in chunks[1:]))
        self.assertTrue(
            all(
                chunk.overlap_token_count <= self.config.overlap_tokens
                for chunk in chunks
            )
        )

    def test_hash_and_metadata_are_stable(self) -> None:
        extraction = process_extracted_pages(["Stable evidence paragraph."])

        first = chunk_extraction(extraction, self.config, self.tokenizer)[0]
        second = chunk_extraction(extraction, self.config, self.tokenizer)[0]

        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.chunking_version, CHUNKER_VERSION)
        self.assertEqual(first.token_count, self.tokenizer.count(first.content))
        self.assertNotIn(first.content_hash, first.content)

    def test_empty_and_tiny_inputs_are_explicit(self) -> None:
        empty = process_extracted_pages([""])
        tiny = process_extracted_pages(["Tiny but meaningful."])

        self.assertEqual(chunk_extraction(empty, self.config, self.tokenizer), [])
        tiny_chunks = chunk_extraction(tiny, self.config, self.tokenizer)
        self.assertEqual(len(tiny_chunks), 1)
        self.assertIn("Tiny but meaningful.", tiny_chunks[0].content)

    def test_configuration_validation_rejects_non_progressing_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than max_tokens"):
            ChunkingConfig(max_tokens=100, overlap_tokens=100).validate()

    def test_evidence_matching_requires_page_and_complete_pattern_group(self) -> None:
        extraction = process_extracted_pages(
            [
                "1 Target Results\n\nTarget amount was 17 percent in FY25.",
                "2 Other Results\n\nUnrelated page.",
            ]
        )
        chunks = chunk_extraction(extraction, self.config, self.tokenizer)
        requirement = {
            "pages": [1],
            "pattern_groups": [["17 percent", "fy25"]],
        }

        self.assertEqual(matching_chunks(chunks, requirement), [0])
        self.assertEqual(
            matching_chunks(
                chunks,
                {"pages": [2], "pattern_groups": [["17 percent", "fy25"]]},
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
