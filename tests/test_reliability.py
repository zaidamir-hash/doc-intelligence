from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from inspect_pdf_extraction import default_sample_pages
from settings import LexisSettings


class ModelReliabilitySettingsTests(unittest.TestCase):
    def test_model_timeout_and_retries_are_environment_configurable(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LEXIS_MODEL_TIMEOUT_SECONDS": "45.5",
                "LEXIS_MODEL_MAX_RETRIES": "3",
            },
        ):
            settings = LexisSettings.from_environment()

        self.assertEqual(settings.model_request_timeout_seconds, 45.5)
        self.assertEqual(settings.model_max_retries, 3)

    def test_invalid_model_request_bounds_are_rejected(self) -> None:
        with patch.dict(os.environ, {"LEXIS_MODEL_TIMEOUT_SECONDS": "0"}):
            with self.assertRaisesRegex(RuntimeError, "positive number"):
                LexisSettings.from_environment()
        with patch.dict(os.environ, {"LEXIS_MODEL_MAX_RETRIES": "-1"}):
            with self.assertRaisesRegex(RuntimeError, "non-negative integer"):
                LexisSettings.from_environment()


class InspectionSamplingTests(unittest.TestCase):
    def test_default_pages_adapt_to_short_and_long_pdfs(self) -> None:
        self.assertEqual(default_sample_pages(1), [1])
        self.assertEqual(default_sample_pages(3), [1, 2, 3])

        long_pdf_pages = default_sample_pages(48)
        self.assertEqual(long_pdf_pages[0], 1)
        self.assertEqual(long_pdf_pages[-1], 48)
        self.assertEqual(long_pdf_pages, sorted(set(long_pdf_pages)))
        self.assertTrue(all(1 <= page <= 48 for page in long_pdf_pages))

    def test_default_pages_reject_invalid_page_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "page_count must be positive"):
            default_sample_pages(0)


if __name__ == "__main__":
    unittest.main()
