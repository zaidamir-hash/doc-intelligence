from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.dataset import DatasetValidationError, load_dataset
from evaluation.metrics import average_metrics, calculate_case_metrics
from evaluation.reporting import render_markdown_report, write_reports
from evaluation.runner import RetrievedChunk, run_evaluation


class DatasetTests(unittest.TestCase):
    def write_dataset(self, directory: str, cases: list[dict[str, object]]) -> Path:
        path = Path(directory) / "dataset.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "test-dataset",
                    "description": "Dataset used by unit tests.",
                    "split": "test",
                    "cases": cases,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_loads_multiple_relevant_chunks_and_unanswerable_case(self) -> None:
        cases = [
            {
                "id": "answerable",
                "filename": "document.pdf",
                "question": "Where is the evidence?",
                "answerable": True,
                "relevant_chunk_indices": [2, 5],
                "relevance_grades": {"2": 2, "5": 1},
                "reference_answer": None,
                "notes": "Two valid passages.",
                "tags": ["paraphrase"],
            },
            {
                "id": "unanswerable",
                "filename": "document.pdf",
                "question": "What is absent?",
                "answerable": False,
                "relevant_chunk_indices": [],
                "relevance_grades": {},
                "reference_answer": None,
                "notes": None,
                "tags": ["unanswerable"],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_dataset(self.write_dataset(directory, cases))

        self.assertEqual(len(dataset.cases), 2)
        self.assertEqual(dataset.cases[0].relevance_grades, {2: 2, 5: 1})
        self.assertFalse(dataset.cases[1].answerable)

    def test_rejects_answerable_case_without_relevant_chunks(self) -> None:
        cases = [
            {
                "id": "invalid",
                "filename": "document.pdf",
                "question": "Where is the evidence?",
                "answerable": True,
                "relevant_chunk_indices": [],
                "tags": [],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(directory, cases)
            with self.assertRaisesRegex(
                DatasetValidationError, "require at least one relevant chunk"
            ):
                load_dataset(path)

    def test_rejects_duplicate_case_ids(self) -> None:
        base_case = {
            "id": "duplicate",
            "filename": "document.pdf",
            "question": "Question",
            "answerable": True,
            "relevant_chunk_indices": [1],
            "tags": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(directory, [base_case, base_case])
            with self.assertRaisesRegex(DatasetValidationError, "must be unique"):
                load_dataset(path)


class MetricTests(unittest.TestCase):
    def test_hit_recall_mrr_and_ndcg(self) -> None:
        metrics = calculate_case_metrics(
            retrieved_chunk_indices=[9, 5, 2, 8],
            relevance_grades={2: 2, 5: 1},
            cutoffs=(1, 3),
        )

        self.assertEqual(metrics.first_relevant_rank, 2)
        self.assertEqual(metrics.reciprocal_rank, 0.5)
        self.assertEqual(metrics.hit_at_k, {1: 0.0, 3: 1.0})
        self.assertEqual(metrics.recall_at_k, {1: 0.0, 3: 1.0})
        self.assertGreater(metrics.ndcg_at_k[3], 0)
        self.assertLessEqual(metrics.ndcg_at_k[3], 1)

    def test_missing_relevant_chunk_scores_zero(self) -> None:
        metrics = calculate_case_metrics([1, 2], {9: 1}, cutoffs=(1, 2))

        self.assertIsNone(metrics.first_relevant_rank)
        self.assertEqual(metrics.reciprocal_rank, 0.0)
        self.assertEqual(metrics.hit_at_k[2], 0.0)
        self.assertEqual(metrics.recall_at_k[2], 0.0)
        self.assertEqual(metrics.ndcg_at_k[2], 0.0)

    def test_averages_questions_equally(self) -> None:
        first = calculate_case_metrics([1], {1: 1}, cutoffs=(1,))
        second = calculate_case_metrics([1], {2: 1}, cutoffs=(1,))

        aggregate = average_metrics([first, second], cutoffs=(1,))

        self.assertEqual(aggregate["hit_at_1"], 0.5)
        self.assertEqual(aggregate["mrr"], 0.5)

    def test_duplicate_retrieval_cannot_receive_relevance_twice(self) -> None:
        metrics = calculate_case_metrics([2, 2], {2: 1}, cutoffs=(2,))

        self.assertEqual(metrics.recall_at_k[2], 1.0)
        self.assertEqual(metrics.ndcg_at_k[2], 1.0)


class RunnerAndReportTests(unittest.TestCase):
    def test_runner_excludes_unanswerable_case_from_aggregate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = DatasetTests().write_dataset(
                directory,
                [
                    {
                        "id": "answerable",
                        "filename": "document.pdf",
                        "question": "Question",
                        "answerable": True,
                        "relevant_chunk_indices": [3],
                        "tags": ["semantic"],
                    },
                    {
                        "id": "unanswerable",
                        "filename": "document.pdf",
                        "question": "Absent question",
                        "answerable": False,
                        "relevant_chunk_indices": [],
                        "tags": ["unanswerable"],
                    },
                ],
            )
            dataset = load_dataset(path)

        def fake_retriever(case, top_k):
            return [
                RetrievedChunk(
                    chunk_id=30,
                    chunk_index=3,
                    filename=case.filename,
                    content="Relevant evidence",
                    distance=0.25,
                )
            ]

        report = run_evaluation(
            dataset,
            fake_retriever,
            top_k=10,
            preview_characters=20,
        )

        self.assertEqual(report["aggregate_metrics"]["hit_at_5"], 1.0)
        self.assertEqual(report["dataset"]["answerable_cases"], 1)
        self.assertEqual(report["dataset"]["unanswerable_cases"], 1)
        self.assertIsNone(report["cases"][1]["metrics"])
        markdown = render_markdown_report(report)
        self.assertIn("First relevant rank", markdown)
        self.assertIn("excluded_from_relevance_metrics", markdown)

        with tempfile.TemporaryDirectory() as output_directory:
            json_path, markdown_path = write_reports(report, Path(output_directory))
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            saved_report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["run"]["run_id"], report["run"]["run_id"])

    def test_empty_dataset_cannot_produce_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = DatasetTests().write_dataset(directory, [])
            dataset = load_dataset(path)

        with self.assertRaisesRegex(ValueError, "contains no cases"):
            run_evaluation(dataset, lambda case, top_k: [], top_k=10)


if __name__ == "__main__":
    unittest.main()
