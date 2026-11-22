"""Reproducible comparison of dense, BM25, hybrid, cross-encoder, and trust-gated RAG."""

import argparse
import csv
import json
import os
import random
import sys
import time

from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.metrics import evaluate_faithfulness


def answer_with_context(question, context):
    from core.state import State
    from reasoning.generator import generate

    state = State(query=question)
    state.context = "\n".join(context)
    answer = generate(state)
    if not answer:
        raise RuntimeError("The configured LLM returned no answer.")
    return answer


def semantic_similarity(model, answer, reference):
    if not answer or not reference:
        return 0.0
    embeddings = model.encode([answer, reference])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


def answer_context_similarity(model, answer, context):
    """Return an embedding similarity proxy; this is not a factuality or entailment score."""
    if not answer or not context:
        return 0.0
    embeddings = model.encode([answer] + context)
    return float(
        max(cosine_similarity([embeddings[0]], embeddings[1:])[0])
    )


def evaluate(args):
    from core.agent import run_agent
    from core.state import State
    from core.trust_policy import DEFAULT_RELEVANCE_WEIGHT, DEFAULT_THRESHOLD
    from retrieval.baselines import (
        bm25_retrieve,
        hybrid_retrieve,
        cross_encoder_rerank_retrieve,
    )
    from retrieval.critic import evaluate_retrieval
    from retrieval.retriever import retrieve_scored
    from retrieval.store import faiss_store, load_dataset_into_faiss
    from utils.llm import model as embedding_model

    queries_path = os.path.join(BASE_DIR, "dataset", "queries.json")
    with open(queries_path, "r", encoding="utf-8") as query_file:
        queries = json.load(query_file)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer.")
        if args.limit < len(queries):
            queries = random.Random(args.seed).sample(queries, args.limit)
    if not queries:
        raise ValueError("No evaluation queries were selected.")

    load_dataset_into_faiss(os.path.join(BASE_DIR, "dataset", "docs"))
    documents = list(faiss_store.texts)
    if not documents:
        raise RuntimeError("The retrieval corpus is empty.")

    rows = []
    systems = [
        "dense_rag",
        "bm25_rag",
        "hybrid_rag",
        "cross_encoder_rag",
        "relevance_only_dense",
        "critic_only_dense",
        "trust_aware_dense",
    ]

    for query_index, item in enumerate(queries, start=1):
        question = item["question"]
        print(f"[{query_index}/{len(queries)}] {question[:72]}")

        contexts = {}
        retrieval_times = {}
        
        # Dense retrieval
        retrieval_start = time.perf_counter()
        dense_scored = retrieve_scored(question)
        retrieval_times["dense_rag"] = time.perf_counter() - retrieval_start
        contexts["dense_rag"] = [text for text, _ in dense_scored]

        # Baseline retrievers
        for system, retriever in (
            ("bm25_rag", lambda: bm25_retrieve(question, documents)),
            ("hybrid_rag", lambda: hybrid_retrieve(question, documents)),
            ("cross_encoder_rag", lambda: cross_encoder_rerank_retrieve(question, documents)),
        ):
            retrieval_start = time.perf_counter()
            contexts[system] = retriever()
            retrieval_times[system] = time.perf_counter() - retrieval_start

        for system in systems:
            state = State(query=question)
            system_retrieval_time = retrieval_times.get(system, 0.0)
            system_start = time.perf_counter()
            generation_start = time.perf_counter()
            
            if system == "trust_aware_dense":
                agent_result = run_agent(state)
                answer = agent_result[0]["text"] if agent_result else ""
                context = state.context.splitlines() if state.context else []
                system_retrieval_time = state.retrieval_time or 0.0
                critic_time = state.critic_time or 0.0
                generation_time = state.generation_time or 0.0
                total_time = time.perf_counter() - system_start
            elif system in ("relevance_only_dense", "critic_only_dense"):
                context = contexts["dense_rag"]
                state.context = "\n".join(context)
                state.retrieval_relevance = max(
                    (max(0.0, min(1.0, score)) for _, score in dense_scored),
                    default=0.0,
                )
                if system == "relevance_only_dense":
                    state.abstained = (
                        not dense_scored
                        or state.retrieval_relevance < DEFAULT_THRESHOLD
                    )
                else:
                    critic_start = time.perf_counter()
                    state.retrieval_verdict = evaluate_retrieval(
                        question,
                        state.context,
                    )
                    state.critic_time = time.perf_counter() - critic_start
                    state.abstained = state.retrieval_verdict == "BAD"
                if state.abstained:
                    answer = ""
                    generation_time = 0.0
                else:
                    generation_start = time.perf_counter()
                    answer = answer_with_context(question, context)
                    generation_time = time.perf_counter() - generation_start
                critic_time = state.critic_time or 0.0
                total_time = (
                    system_retrieval_time + critic_time + generation_time
                )
            else:
                context = contexts[system]
                answer = answer_with_context(question, context)
                generation_time = time.perf_counter() - generation_start
                critic_time = 0.0
                total_time = system_retrieval_time + generation_time

            # Compute claim-level groundedness / faithfulness
            faithfulness_dict = evaluate_faithfulness(answer, "\n".join(context))

            rows.append({
                "query_id": item.get("id", query_index),
                "question": question,
                "system": system,
                "answer": answer,
                "reference_answer": item.get("ground_truth", ""),
                "semantic_similarity": semantic_similarity(
                    embedding_model,
                    answer,
                    item.get("ground_truth", ""),
                ),
                "answer_context_similarity_proxy": answer_context_similarity(
                    embedding_model,
                    answer,
                    context,
                ),
                "faithfulness_score": faithfulness_dict["faithfulness_score"],
                "hallucination_rate": faithfulness_dict["hallucination_rate"],
                "num_claims": faithfulness_dict["num_claims"],
                "unsupported_claims": faithfulness_dict["unsupported_claims"],
                "refusal": int(
                    state.abstained
                    or "insufficient information" in answer.lower()
                ),
                "retrieved_context": "\n---\n".join(context),
                "trust_score": state.trust_score if system == "trust_aware_dense" else "",
                "retrieval_verdict": (
                    state.retrieval_verdict
                    if system in ("critic_only_dense", "trust_aware_dense")
                    else ""
                ),
                "retrieval_relevance": (
                    state.retrieval_relevance
                    if system in (
                        "relevance_only_dense",
                        "critic_only_dense",
                        "trust_aware_dense",
                    )
                    else ""
                ),
                "trust_threshold": (
                    DEFAULT_THRESHOLD if system == "trust_aware_dense" else ""
                ),
                "trust_relevance_weight": (
                    DEFAULT_RELEVANCE_WEIGHT if system == "trust_aware_dense" else ""
                ),
                "run_seed": args.seed,
                "embedding_model": "all-MiniLM-L6-v2",
                "generator_model": "llama3",
                "retrieval_time_seconds": system_retrieval_time,
                "critic_time_seconds": critic_time,
                "generation_time_seconds": generation_time,
                "total_time_seconds": total_time,
                "llm_call_count": (
                    int(bool(state.context))
                    + int(bool(answer))
                    if system == "trust_aware_dense"
                    else 1
                ),
            })

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} system-query rows to {output_path}")
    print(
        "Note: Includes claim-level faithfulness and hallucination rates in addition to "
        "semantic similarity and timing metrics."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Number of dataset rows to evaluate; use 1000 for the full dataset.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Recorded run seed.")
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "evaluation", "revision_results.csv"),
        help="Output CSV; the existing results.csv is never overwritten by default.",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
