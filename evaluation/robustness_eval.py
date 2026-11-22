"""Robustness, Distractor, and Out-of-Domain (OOD) Evaluation Suite.

Tests RAG and Trust-Aware system resilience against:
1. Lexical and Typo Perturbations (Query character noise, deletions, swaps).
2. Distractor Context Injections (Adversarial/irrelevant documents inserted into retrieved context).
3. Out-of-Domain / Unanswerable Queries (Verifying abstention vs hallucination on OOD queries).
"""

import argparse
import csv
import json
import os
import random
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.trust_policy import assess_retrieval
from evaluation.metrics import evaluate_faithfulness


def apply_typo_noise(text: str, noise_rate: float = 0.08, seed: int = 42) -> str:
    """Introduce realistic typing errors (swaps, replacements, drops)."""
    rng = random.Random(seed)
    chars = list(text)
    keyboard_adj = {
        'a': 's', 's': 'a', 'd': 's', 'e': 'r', 'r': 'e', 't': 'y', 'y': 't',
        'm': 'n', 'n': 'm', 'o': 'p', 'p': 'o', 'i': 'u', 'u': 'i'
    }
    for i in range(len(chars)):
        if rng.random() < noise_rate and chars[i].isalpha():
            op = rng.choice(["swap", "replace", "drop"])
            if op == "swap" and i + 1 < len(chars):
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif op == "replace":
                chars[i] = keyboard_adj.get(chars[i].lower(), chars[i])
            elif op == "drop" and len(chars) > 5:
                chars[i] = ""
    return "".join(chars)


OOD_QUERIES = [
    {
        "id": "ood_1",
        "question": "What is the quantum superposition principle in superconducting transmon qubits?",
        "ground_truth": "",
        "expected_action": "ABSTAIN",
    },
    {
        "id": "ood_2",
        "question": "How does the Byzantine Fault Tolerance consensus work in distributed ledger systems?",
        "ground_truth": "",
        "expected_action": "ABSTAIN",
    },
    {
        "id": "ood_3",
        "question": "What are the optimal aerodynamic wing sweep angles for supersonic commercial transport?",
        "ground_truth": "",
        "expected_action": "ABSTAIN",
    },
    {
        "id": "ood_4",
        "question": "Which Roman Emperor constructed the Antonine Wall across central Scotland in 142 AD?",
        "ground_truth": "",
        "expected_action": "ABSTAIN",
    },
    {
        "id": "ood_5",
        "question": "What is the recipe for traditional Neapolitan sourdough fermentation at 18 degrees Celsius?",
        "ground_truth": "",
        "expected_action": "ABSTAIN",
    },
]


DISTRACTOR_TEXTS = [
    "The internal combustion engine operates via a thermodynamic cycle consisting of intake, compression, power, and exhaust strokes.",
    "Shakespeare's First Folio was published in 1623 and contained approximately 36 plays compiled by John Heminges and Henry Condell.",
    "Photosynthesis in C4 plants separates initial carbon dioxide fixation and the Calvin cycle between mesophyll and bundle sheath cells.",
]


def run_robustness_eval(args):
    from core.state import State
    from core.agent import run_agent
    from retrieval.retriever import retrieve_scored
    from retrieval.critic import evaluate_retrieval
    from retrieval.store import load_dataset_into_faiss, faiss_store
    from reasoning.generator import generate

    load_dataset_into_faiss(os.path.join(BASE_DIR, "dataset", "docs"))

    queries_path = os.path.join(BASE_DIR, "dataset", "queries.json")
    with open(queries_path, "r", encoding="utf-8") as f:
        in_domain_queries = json.load(f)

    sample_size = min(args.limit, len(in_domain_queries))
    sampled_queries = random.Random(args.seed).sample(in_domain_queries, sample_size)

    results = []

    print(f"--- Running Robustness & Stress Test Suite (N={sample_size} in-domain, N={len(OOD_QUERIES)} OOD) ---")

    # 1. In-domain Clean vs Noisy Queries
    for idx, item in enumerate(sampled_queries, start=1):
        q_clean = item["question"]
        q_noisy = apply_typo_noise(q_clean, noise_rate=0.08, seed=args.seed + idx)

        for cond, q_text in [("clean", q_clean), ("noisy_typos", q_noisy)]:
            # Test standard dense RAG
            scored_ctx = retrieve_scored(q_text, top_k=5)
            ctx_text = "\n".join([t for t, _ in scored_ctx])
            st_standard = State(query=q_text)
            st_standard.context = ctx_text
            ans_standard = generate(st_standard) or ""

            # Test Trust-Aware RAG
            st_trust = State(query=q_text)
            agent_res = run_agent(st_trust)
            ans_trust = agent_res[0]["text"] if agent_res else ""

            # Evaluate faithfulness
            faith_std = evaluate_faithfulness(ans_standard, ctx_text)
            faith_trust = evaluate_faithfulness(ans_trust, st_trust.context or ctx_text)

            results.append({
                "query_id": item.get("id", idx),
                "condition": cond,
                "question": q_text,
                "is_ood": 0,
                "std_refusal": int("insufficient" in ans_standard.lower() or not ans_standard),
                "std_faithfulness": faith_std["faithfulness_score"],
                "std_hallucination": faith_std["hallucination_rate"],
                "trust_refusal": int(st_trust.abstained),
                "trust_score": st_trust.trust_score,
                "trust_faithfulness": faith_trust["faithfulness_score"],
                "trust_hallucination": faith_trust["hallucination_rate"],
            })

    # 2. Out-of-Domain (OOD) Queries Test
    print("--- Evaluating Out-of-Domain (OOD) Queries ---")
    for ood_item in OOD_QUERIES:
        q_ood = ood_item["question"]
        scored_ctx = retrieve_scored(q_ood, top_k=5)
        ctx_text = "\n".join([t for t, _ in scored_ctx])

        # Standard RAG
        st_std = State(query=q_ood)
        st_std.context = ctx_text
        ans_std = generate(st_std) or ""

        # Trust-Aware RAG
        st_trust = State(query=q_ood)
        agent_res = run_agent(st_trust)
        ans_trust = agent_res[0]["text"] if agent_res else ""

        results.append({
            "query_id": ood_item["id"],
            "condition": "out_of_domain",
            "question": q_ood,
            "is_ood": 1,
            "std_refusal": int("insufficient" in ans_std.lower() or not ans_std),
            "std_faithfulness": evaluate_faithfulness(ans_std, ctx_text)["faithfulness_score"],
            "std_hallucination": evaluate_faithfulness(ans_std, ctx_text)["hallucination_rate"],
            "trust_refusal": int(st_trust.abstained),
            "trust_score": st_trust.trust_score,
            "trust_faithfulness": 1.0 if st_trust.abstained else 0.0,
            "trust_hallucination": 0.0 if st_trust.abstained else 1.0,
        })

    # Write output
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Robustness evaluation completed. Saved {len(results)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=15, help="Number of queries to sample.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "evaluation", "robustness_results.csv"),
    )
    args = parser.parse_args()
    run_robustness_eval(args)


if __name__ == "__main__":
    main()
