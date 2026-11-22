import unittest
import csv
import os
import tempfile

from core.trust_policy import assess_retrieval
from retrieval.baselines import bm25_retrieve, cross_encoder_rerank_retrieve, hybrid_retrieve
from evaluation.create_human_eval import create_annotation_files
from evaluation.metrics import (
    evaluate_faithfulness,
    split_into_claims,
    compute_selective_prediction_metrics,
    bootstrap_confidence_interval,
)
from evaluation.robustness_eval import apply_typo_noise
from evaluation.failure_analysis import categorize_query_failure


class TrustPolicyTests(unittest.TestCase):
    def test_bad_verdict_always_abstains(self):
        assessment = assess_retrieval([0.99], "BAD")
        self.assertTrue(assessment.abstain)
        self.assertEqual(assessment.trust_score, 0.495)

    def test_partial_verdict_requires_relevant_retrieval(self):
        self.assertTrue(assess_retrieval([0.49], "PARTIAL").abstain)
        self.assertFalse(assess_retrieval([0.6], "PARTIAL").abstain)

    def test_good_verdict_passes_and_scores_are_bounded(self):
        assessment = assess_retrieval([1.2], "GOOD")
        self.assertFalse(assessment.abstain)
        self.assertEqual(assessment.trust_score, 1.0)

    def test_unknown_verdict_is_an_error(self):
        with self.assertRaises(ValueError):
            assess_retrieval([0.8], "UNKNOWN")


class RetrievalBaselineTests(unittest.TestCase):
    def test_bm25_returns_most_relevant_documents_first(self):
        documents = [
            "mitochondria and programmed cell death",
            "snellen visual acuity measurement",
            "mitochondria support cell energy",
        ]
        results = bm25_retrieve("mitochondria cell", documents, top_k=2)
        self.assertEqual(results[0], documents[2])
        self.assertEqual(len(results), 2)

    def test_bm25_returns_no_results_without_overlap(self):
        self.assertEqual(
            bm25_retrieve("quantum computing", ["eye acuity"], top_k=5),
            [],
        )

    def test_hybrid_and_cross_encoder_fallbacks(self):
        documents = [
            "acute coronary syndrome guidelines and intervention",
            "pediatric asthma emergency management",
        ]
        res = cross_encoder_rerank_retrieve("coronary intervention", documents, top_k=1)
        self.assertTrue(len(res) <= 1)


class MetricsAndFaithfulnessTests(unittest.TestCase):
    def test_claim_splitting_and_faithfulness(self):
        context = "Patients treated with metformin showed reduced fasting blood glucose and HbA1c levels."
        grounded_answer = "Metformin reduces fasting blood glucose."
        ungrounded_answer = "Metformin causes immediate severe neurological blindness and heart failure."

        f_grounded = evaluate_faithfulness(grounded_answer, context)
        self.assertGreaterEqual(f_grounded["faithfulness_score"], 0.7)
        self.assertLessEqual(f_grounded["hallucination_rate"], 0.3)

        f_ungrounded = evaluate_faithfulness(ungrounded_answer, context)
        self.assertGreater(f_ungrounded["hallucination_rate"], 0.5)

    def test_selective_prediction_metrics(self):
        preds = ["A", "B", "C", "D"]
        refs = ["A", "B", "X", "Y"]
        abst = [False, False, True, True]
        metrics = compute_selective_prediction_metrics(preds, refs, abst)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["answered_queries"], 2)

    def test_bootstrap_ci(self):
        values = [0.8, 0.85, 0.9, 0.82, 0.88]
        mean, low, high = bootstrap_confidence_interval(values, num_samples=100)
        self.assertTrue(low <= mean <= high)


class RobustnessAndFailureTests(unittest.TestCase):
    def test_typo_noise_generation(self):
        text = "cardiovascular diseases treatment"
        noisy = apply_typo_noise(text, noise_rate=0.2, seed=42)
        self.assertNotEqual(text, noisy)
        self.assertTrue(len(noisy) > 10)

    def test_failure_categorization(self):
        row_miss = {
            "answer": "",
            "refusal": 1,
            "retrieval_relevance": 0.2,
            "retrieval_verdict": "BAD",
        }
        cat, expl = categorize_query_failure(row_miss)
        self.assertEqual(cat, "Type I: Retrieval Miss")


class HumanEvaluationSheetTests(unittest.TestCase):
    def test_pairwise_sheet_blinds_system_names_and_writes_separate_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "results.csv")
            output_path = os.path.join(temp_dir, "human_eval.csv")
            with open(input_path, "w", encoding="utf-8", newline="") as source:
                writer = csv.DictWriter(
                    source,
                    fieldnames=[
                        "query_id",
                        "question",
                        "system",
                        "answer",
                        "retrieved_context",
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "query_id": "q1",
                        "question": "Question?",
                        "system": "dense_rag",
                        "answer": "Answer one",
                        "retrieved_context": "Evidence one",
                    },
                    {
                        "query_id": "q1",
                        "question": "Question?",
                        "system": "bm25_rag",
                        "answer": "Answer two",
                        "retrieved_context": "Evidence two",
                    },
                ])

            create_annotation_files(input_path, output_path, seed=13)

            with open(output_path, "r", encoding="utf-8", newline="") as result:
                annotation = next(csv.DictReader(result))
            self.assertNotIn("system", annotation)
            self.assertEqual(annotation["correctness_preference"], "")

            with open(
                os.path.join(temp_dir, "human_eval_key.csv"),
                "r",
                encoding="utf-8",
                newline="",
            ) as key_file:
                key_rows = list(csv.DictReader(key_file))
            self.assertEqual({row["system"] for row in key_rows}, {"dense_rag", "bm25_rag"})


if __name__ == "__main__":
    unittest.main()
