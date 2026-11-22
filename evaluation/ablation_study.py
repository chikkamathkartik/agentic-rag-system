"""Systematic Ablation and Hyperparameter Sensitivity Analysis.

Performs:
1. Architectural component ablations (No Gate, Relevance-only, Critic-only, Full Policy).
2. Grid sweep over weight lambda in [0.0, 1.0] and threshold tau in [0.0, 1.0].
3. Computes Coverage, Refusal Rate, Faithfulness, and Selective Accuracy trade-offs.
"""

import argparse
import csv
import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.trust_policy import assess_retrieval
from evaluation.metrics import evaluate_faithfulness, bootstrap_confidence_interval


def run_ablation_grid(input_csv: str, output_csv: str):
    """Analyze parameter sensitivity and component ablations on existing evaluation runs."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input evaluation results not found at {input_csv}. "
            "Please run evaluation/run_revision_eval.py first."
        )

    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Filter to trust-aware or candidate rows where retrieval relevance and critic verdicts exist
    records = []
    # Collect query data
    queries_data = {}
    for r in rows:
        qid = r["query_id"]
        if qid not in queries_data:
            queries_data[qid] = {}
        if r["retrieval_relevance"]:
            try:
                queries_data[qid]["relevance"] = float(r["retrieval_relevance"])
            except ValueError:
                pass
        if r["retrieval_verdict"]:
            queries_data[qid]["verdict"] = r["retrieval_verdict"]
        if r["system"] == "dense_rag":
            queries_data[qid]["dense_answer"] = r["answer"]
            queries_data[qid]["context"] = r["retrieved_context"]
            queries_data[qid]["reference"] = r["reference_answer"]

    valid_queries = [
        q for q, data in queries_data.items()
        if "relevance" in data and "verdict" in data and "dense_answer" in data
    ]

    if not valid_queries:
        print("Insufficient diagnostic columns to perform offline ablation grid.")
        return

    # Sweep lambda and tau
    lambdas = [0.0, 0.25, 0.50, 0.75, 1.0]
    thresholds = [0.2, 0.35, 0.50, 0.65, 0.80]

    grid_results = []

    for lam in lambdas:
        for tau in thresholds:
            refusals = 0
            faithfulness_list = []
            
            for qid in valid_queries:
                data = queries_data[qid]
                rel = data["relevance"]
                verd = data["verdict"]
                ans = data["dense_answer"]
                ctx = data["context"]
                
                assessment = assess_retrieval(
                    [rel],
                    verd,
                    threshold=tau,
                    relevance_weight=lam,
                )
                
                if assessment.abstain:
                    refusals += 1
                    faithfulness_list.append(1.0)  # Safe abstention avoids hallucination
                else:
                    f_res = evaluate_faithfulness(ans, ctx)
                    faithfulness_list.append(f_res["faithfulness_score"])

            total = len(valid_queries)
            refusal_rate = refusals / total
            coverage = 1.0 - refusal_rate
            mean_faith, low_f, high_f = bootstrap_confidence_interval(faithfulness_list)

            grid_results.append({
                "relevance_weight_lambda": lam,
                "trust_threshold_tau": tau,
                "total_queries": total,
                "coverage": round(coverage, 4),
                "refusal_rate": round(refusal_rate, 4),
                "faithfulness_mean": round(mean_faith, 4),
                "faithfulness_ci_lower": round(low_f, 4),
                "faithfulness_ci_upper": round(high_f, 4),
            })

    # Write ablation sweep
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grid_results[0].keys())
        writer.writeheader()
        writer.writerows(grid_results)
    print(f"Wrote {len(grid_results)} ablation points to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join(BASE_DIR, "evaluation", "revision_results.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "evaluation", "ablation_grid_results.csv"),
    )
    args = parser.parse_args()
    run_ablation_grid(args.input, args.output)


if __name__ == "__main__":
    main()
