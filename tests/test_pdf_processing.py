from __future__ import annotations

import unittest

from pdf_processing import (
    PAGE_LABEL_TEMPLATE,
    chunk_extracted_pages,
    chunk_text,
    process_extracted_pages,
)


class PdfCleaningTests(unittest.TestCase):
    def test_preserves_page_numbers_and_paragraph_boundaries(self) -> None:
        result = process_extracted_pages(
            ["First paragraph.\n\nSecond   paragraph.", "Page two text."]
        )

        self.assertEqual([page.page_number for page in result.pages], [1, 2])
        self.assertEqual(
            result.pages[0].cleaned_text,
            "First paragraph.\n\nSecond paragraph.",
        )

    def test_repairs_only_conservative_wrapped_prose(self) -> None:
        result = process_extracted_pages(
            [
                "This deliberately long prose line continues\n"
                "with a lower-case wrapped continuation.\n\n"
                "1. A list item\n"
                "2. Another list item\n\n"
                "climate-resil-\nient farming"
            ]
        )

        cleaned = result.pages[0].cleaned_text
        self.assertIn(
            "This deliberately long prose line continues with a lower-case "
            "wrapped continuation.",
            cleaned,
        )
        self.assertIn("1. A list item\n2. Another list item", cleaned)
        self.assertIn("climate-resilient farming", cleaned)

    def test_removes_only_repeated_page_edge_furniture(self) -> None:
        raw_pages = [
            f"Annual Report 2024-25\nUnique heading {number}\nBody for page {number}.\n{number}"
            for number in range(1, 6)
        ]

        result = process_extracted_pages(raw_pages)

        self.assertEqual(result.repeated_headers, ("Annual Report 2024-25",))
        self.assertEqual(result.repeated_footers, ("1",))
        self.assertIn("Annual Report 2024-25", result.pages[0].cleaned_text)
        for page in result.pages[1:]:
            self.assertNotIn("Annual Report 2024-25", page.cleaned_text)
            self.assertEqual(len(page.removed_headers), 1)
            self.assertEqual(len(page.removed_footers), 1)
            self.assertIn("Unique heading", page.cleaned_text)

    def test_empty_and_low_text_pages_have_visible_warnings(self) -> None:
        result = process_extracted_pages(["", "Tiny"])

        self.assertEqual(result.pages[0].warnings[0].code, "empty-page")
        self.assertEqual(result.pages[1].warnings[0].code, "low-text-page")

    def test_removes_pdf_control_characters(self) -> None:
        result = process_extracted_pages(["Heading\n\x03\nUseful body text."])

        self.assertNotIn("\x03", result.pages[0].cleaned_text)
        self.assertIn("Useful body text.", result.pages[0].cleaned_text)

    def test_deduplicates_exact_long_paragraphs_but_keeps_short_values(self) -> None:
        paragraph = "A long duplicated extraction-layer paragraph. " * 3
        result = process_extracted_pages(
            [f"{paragraph}\n\n{paragraph}\n\n25\n\n25"]
        )

        self.assertEqual(result.pages[0].cleaned_text.count(paragraph.strip()), 1)
        self.assertEqual(result.pages[0].cleaned_text.count("25"), 2)


class PageChunkTests(unittest.TestCase):
    def test_every_chunk_has_visible_page_provenance(self) -> None:
        result = process_extracted_pages(["A" * 120, "B" * 120])
        chunks = chunk_extracted_pages(result, chunk_size=80, overlap=20)

        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            expected_label = PAGE_LABEL_TEMPLATE.format(
                page_number=chunk.page_start
            )
            self.assertTrue(chunk.content.startswith(expected_label))
            self.assertEqual(chunk.page_start, chunk.page_end)

    def test_chunk_configuration_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than chunk_size"):
            chunk_text("text", chunk_size=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
