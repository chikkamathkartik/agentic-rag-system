"""Automated and Qualitative Failure Mode Analysis for Trust-Aware RAG.

Classifies failure cases into an explicit five-class taxonomy:
  Type I: Retrieval Miss (Low relevance, relevant facts missing from top-k context)
  Type II: Incomplete Evidence / Sub-threshold Marginality (Partial context, trust < tau)
  Type III: Critic Misjudgment (Critic verdict discordant with context content)
  Type IV: Generator Hallucination (Grounded context present, but model hallucinates unsupported claims)
  Type V: Over-cautious Abstention (System refrains on queries answerable from context)
"""

import argparse
import csv
import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.metrics import evaluate_faithfulness


def categorize_query_failure(row: dict) -> tuple[str, str]:
    """Diagnose the primary failure mode for a given query evaluation record."""
    answer = row.get("answer", "")
    refusal = int(row.get("refusal", 0)) == 1
    rel = float(row.get("retrieval_relevance", 0.0) or 0.0)
    verdict = row.get("retrieval_verdict", "").strip().upper()
    sim = float(row.get("semantic_similarity", 0.0) or 0.0)
    context = row.get("retrieved_context", "")

    # Check if answer was generated
    if refusal or not answer:
        if rel < 0.35 and verdict == "BAD":
            return (
                "Type I: Retrieval Miss",
                f"Retriever failed to fetch relevant context (relevance={rel:.2f}, verdict={verdict}). Safely abstained."
            )
        elif verdict == "PARTIAL":
            return (
                "Type II: Incomplete Evidence Abstention",
                f"Retrieved passages provided partial coverage but combined trust was sub-threshold. System conservatively abstained."
            )
        elif verdict == "BAD" and rel > 0.6:
            return (
                "Type III: Critic False Negative",
                f"Retriever found high-similarity text (relevance={rel:.2f}), but critic rejected it as BAD."
            )
        elif verdict == "GOOD":
            return (
                "Type V: Over-cautious Abstention",
                f"Critic found context GOOD, but system triggered refusal fallback."
            )
        else:
            return (
                "Type I: Retrieval Miss",
                f"Low retrieval signal (relevance={rel:.2f}). System abstained."
            )
    else:
        # Answer was produced - evaluate hallucination / correctness
        faith_res = evaluate_faithfulness(answer, context)
        if faith_res["hallucination_rate"] > 0.4:
            return (
                "Type IV: Generator Ungrounded Hallucination",
                f"Generated answer contains unsupported claims (hallucination_rate={faith_res['hallucination_rate']:.2f}, unsupported={faith_res['unsupported_claims']}/{faith_res['num_claims']})."
            )
        elif sim < 0.4:
            return (
                "Type III: Critic False Positive / Semantic Mismatch",
                f"Answer was generated but semantic similarity to reference is low (similarity={sim:.2f})."
            )
        else:
            return ("Success", "Faithful and semantically aligned answer.")


def analyze_failures(input_csv: str, output_report: str):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    systems = sorted({r["system"] for r in rows})
    failure_counts = {sys: {} for sys in systems}
    case_studies = []

    for r in rows:
        sys_name = r["system"]
        failure_type, explanation = categorize_query_failure(r)
        failure_counts[sys_name][failure_type] = failure_counts[sys_name].get(failure_type, 0) + 1

        if failure_type != "Success" and len(case_studies) < 15:
            case_studies.append({
                "system": sys_name,
                "query_id": r["query_id"],
                "question": r["question"][:100],
                "failure_type": failure_type,
                "explanation": explanation,
                "answer_snippet": r["answer"][:120] if r["answer"] else "[ABSTAINED]",
                "relevance": r.get("retrieval_relevance", "N/A"),
                "verdict": r.get("retrieval_verdict", "N/A"),
            })

    # Generate markdown report
    md_lines = [
        "# Comprehensive Failure Mode Taxonomy and Diagnostic Report",
        "",
        "This report provides an empirical breakdown of failure modes observed during comparative evaluation.",
        "",
        "## 1. Five-Class Failure Taxonomy",
        "",
        "| Failure Mode | Definition | Root Cause | System Behavior |",
        "| :--- | :--- | :--- | :--- |",
        "| **Type I: Retrieval Miss** | Top-$k$ retrieved chunks contain negligible relevant facts | Query-document embedding mismatch or missing corpus documents | Appropriately detected by Critic/Trust; system abstains |",
        "| **Type II: Incomplete Evidence** | Chunks partially overlap question, but key premises are omitted | Document segmentation or incomplete evidence | Critic yields `PARTIAL`; Trust score $< \\tau$; system abstains |",
        "| **Type III: Critic Misjudgment** | Critic verdict diverges from true informational context | LLM prompt sensitivity or semantic ambiguity | False positive leads to generation; False negative causes refusal |",
        "| **Type IV: Generator Hallucination** | Answer asserts facts not grounded in retrieved context | Autoregressive prior overrides context | Faithfulness drop detected by claim-level verification |",
        "| **Type V: Over-cautious Refusal** | Abstention triggered on borderline answerable question | High conservatism threshold ($\\tau$) | Coverage drops despite answer feasibility |",
        "",
        "## 2. Quantitative Failure Mode Distribution Across Systems",
        "",
    ]

    all_failure_keys = [
        "Success",
        "Type I: Retrieval Miss",
        "Type II: Incomplete Evidence Abstention",
        "Type III: Critic False Negative",
        "Type III: Critic False Positive / Semantic Mismatch",
        "Type IV: Generator Ungrounded Hallucination",
        "Type V: Over-cautious Abstention",
    ]

    header = "| Failure Class | " + " | ".join(systems) + " |"
    sep = "| :--- | " + " | ".join([":---:"] * len(systems)) + " |"
    md_lines.append(header)
    md_lines.append(sep)

    for k in all_failure_keys:
        row_str = f"| **{k}** |"
        for sys_name in systems:
            count = failure_counts[sys_name].get(k, 0)
            row_str += f" {count} |"
        md_lines.append(row_str)

    md_lines.extend([
        "",
        "## 3. Representative Case Studies",
        "",
    ])

    for i, cs in enumerate(case_studies, start=1):
        md_lines.extend([
            f"### Case {i}: [{cs['failure_type']}] (System: `{cs['system']}`)",
            f"- **Query ID**: `{cs['query_id']}`",
            f"- **Question**: {cs['question']}",
            f"- **Diagnostics**: Relevance: `{cs['relevance']}`, Verdict: `{cs['verdict']}`",
            f"- **Output**: *{cs['answer_snippet']}*",
            f"- **Diagnosis**: {cs['explanation']}",
            "",
        ])

    out_file = os.path.abspath(output_report)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Detailed failure analysis written to {out_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join(BASE_DIR, "evaluation", "revision_results.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "evaluation", "failure_analysis_report.md"),
    )
    args = parser.parse_args()
    analyze_failures(args.input, args.output)


if __name__ == "__main__":
    main()
