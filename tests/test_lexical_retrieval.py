from __future__ import annotations

import os
import unittest

from database import SessionLocal
from lexical_retrieval import parse_lexical_query, retrieve_lexical_candidates


class LexicalQueryParsingTests(unittest.TestCase):
    def test_plain_question_terms_are_joined_with_or(self) -> None:
        parsed = parse_lexical_query("What does PRISM+ stand for?")

        self.assertEqual(parsed.search_text, "What OR does OR PRISM+ OR stand OR for")
        self.assertEqual(parsed.exact_terms, ("PRISM+",))

    def test_explicit_phrase_remains_a_phrase(self) -> None:
        parsed = parse_lexical_query('What is "Climate Risk Fund-I" for?')

        self.assertTrue(parsed.search_text.startswith('"Climate Risk Fund-I" OR'))
        self.assertIn("Climate Risk Fund-I", parsed.exact_terms)

    def test_acronyms_and_punctuation_identifiers_are_inspectable(self) -> None:
        parsed = parse_lexical_query(
            "Did AAOIFI use Section 39(1) during FY2024-25?"
        )

        self.assertEqual(parsed.exact_terms, ("AAOIFI", "39(1)", "FY2024-25"))

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            parse_lexical_query("  ")


@unittest.skipUnless(
    os.getenv("RUN_DATABASE_TESTS") == "1",
    "set RUN_DATABASE_TESTS=1 to run PostgreSQL integration tests",
)
class LexicalPostgresTests(unittest.TestCase):
    def test_ranked_results_expose_fts_and_exact_diagnostics(self) -> None:
        database = SessionLocal()
        try:
            results = retrieve_lexical_candidates(
                "What does the acronym PRISM+ stand for?",
                database,
                "Gov-AR_1 test.pdf",
                candidate_k=10,
            )
        finally:
            database.close()

        self.assertTrue(results)
        self.assertEqual([result.lexical_rank for result in results], list(range(1, len(results) + 1)))
        self.assertTrue(all(result.lexical_score >= 0 for result in results))
        self.assertTrue(any("PRISM+" in result.exact_terms_matched for result in results))
        self.assertTrue(any(result.chunk_index in {20, 420} for result in results[:5]))

    def test_exact_signal_categories_retrieve_expected_evidence(self) -> None:
        cases = (
            ('What is "Climate Risk Fund-I" intended to achieve?', {18}),
            ("How many AAOIFI standards were adopted during FY2024-25?", {157}),
            ("What does PRISM+ stand for?", {20, 420}),
            (
                "Under which statutory provision was the Governor's Annual "
                "Report submitted to Parliament?",
                {5, 6},
            ),
            ("What NFIS 2024-28 targets were set for 2028?", {22, 337, 338}),
        )
        database = SessionLocal()
        try:
            for question, expected_indices in cases:
                with self.subTest(question=question):
                    results = retrieve_lexical_candidates(
                        question,
                        database,
                        "Gov-AR_1 test.pdf",
                        candidate_k=5,
                    )
                    self.assertTrue(
                        expected_indices & {result.chunk_index for result in results}
                    )
        finally:
            database.close()

    def test_common_word_only_query_fails_safely_with_no_results(self) -> None:
        database = SessionLocal()
        try:
            results = retrieve_lexical_candidates(
                "the and or",
                database,
                "Gov-AR_1 test.pdf",
                candidate_k=10,
            )
        finally:
            database.close()

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
