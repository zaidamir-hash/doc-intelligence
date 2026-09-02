from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from evaluation.dataset import load_dataset


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "evaluation" / "experiments" / "final_ablation_report.json"


class FinalLabelledCorpusRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def test_fixed_retrieval_dataset_identity_and_coverage(self) -> None:
        metadata = self.report["retrieval_dataset"]
        path = ROOT / metadata["path"]
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata["sha256"],
        )
        dataset = load_dataset(path)
        self.assertEqual(dataset.split, "test")
        self.assertEqual(len(dataset.cases), 14)
        self.assertEqual(sum(case.answerable for case in dataset.cases), 13)
        self.assertTrue(
            all(
                case.document_content_hash and case.relevant_chunk_hashes
                for case in dataset.cases
                if case.answerable
            )
        )

    def test_fixed_answer_dataset_identity(self) -> None:
        metadata = self.report["answer_dataset"]
        path = ROOT / metadata["path"]
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata["sha256"],
        )

    def test_all_required_ablation_stages_are_sourced(self) -> None:
        stages = {stage["id"]: stage for stage in self.report["retrieval_stages"]}
        required = {
            "original_baseline",
            "clean_extraction_dense",
            "token_structure_dense",
            "dense_final",
            "lexical_final",
            "hybrid_rrf",
            "hybrid_reranked",
            "hybrid_reranked_expanded",
        }
        self.assertEqual(set(stages), required)
        for stage in stages.values():
            source = ROOT / stage["source_artifact"]
            self.assertTrue(source.is_file(), stage["source_artifact"])
            self.assertIn(stage["source_run_id"], source.read_text(encoding="utf-8"))

    def test_final_retrieval_does_not_regress_below_phase1_baseline(self) -> None:
        stages = {stage["id"]: stage for stage in self.report["retrieval_stages"]}
        baseline = stages["original_baseline"]["metrics"]
        final = stages["hybrid_reranked_expanded"]["metrics"]
        for metric in ("hit_at_5", "mrr", "recall_at_5", "ndcg_at_5"):
            self.assertGreaterEqual(final[metric], baseline[metric], metric)

    def test_answer_and_retrieval_results_remain_separate(self) -> None:
        answer = self.report["answer_evaluation"]
        self.assertIn("retrieval_success", answer)
        self.assertIn("answer_success", answer)
        self.assertEqual(answer["retrieval_success"], 1.0)
        self.assertEqual(answer["unsupported_claim_rate"], 0.0)
        self.assertLess(answer["answer_success"], answer["retrieval_success"])


if __name__ == "__main__":
    unittest.main()
