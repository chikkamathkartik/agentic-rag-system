import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PATH ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# ---------- IMPORTS ----------
from baseline import baseline_rag
from core.agent import run_agent
from core.state import State
from retrieval.retriever import retrieve

# ---------- LOAD DATA ----------
queries_path = os.path.join(BASE_DIR, "dataset", "queries.json")

with open(queries_path, "r") as f:
    queries = json.load(f)

queries = queries[:30]

OOD_QUERIES = [
    "What is quantum computing?",
    "Explain blockchain",
    "What is relativity?"
]

# ---------- MODEL ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- SIMILARITY ----------
def similarity(a, b):
    if not a or not b:
        return 0
    emb1 = model.encode([a])
    emb2 = model.encode([b])
    return cosine_similarity(emb1, emb2)[0][0]

# ---------- HALLUCINATION ----------
def is_hallucinated(answer, chunks):

    if not answer:
        return 0

    if "insufficient information" in answer.lower():
        return 0

    if not chunks:
        return 1

    scores = [similarity(answer, c) for c in chunks]
    return 1 if max(scores) < 0.15 else 0



# ---------- EVALUATION ----------
results = []

for i, q in enumerate(queries):

    question = q["question"]
    ground_truth = q["ground_truth"]

    print(f"\n[{i+1}/{len(queries)}] Question: {question[:60]}...")

    # ---------- BASELINE ----------
    try:
        baseline_start = time.time()
        baseline_answer = baseline_rag(question) or ""
        baseline_time = time.time() - baseline_start
    except Exception as e:
        print("Baseline error:", e)
        baseline_answer = ""
        baseline_time = 0

    # ---------- AGENT ----------
    state = State(query=question)
    agent_start = time.time()
    memory = run_agent(state)
    agent_total_time = time.time() - agent_start
    
    agent_answer = memory[0]["text"] if memory else ""

    # ---------- CONTEXT ----------
    context_chunks = retrieve(question)

    # ---------- METRICS ----------
    sim_baseline = similarity(baseline_answer, ground_truth)
    sim_agent = similarity(agent_answer, ground_truth)

    hallucination_baseline = is_hallucinated(baseline_answer, context_chunks)
    hallucination_agent = is_hallucinated(agent_answer, context_chunks)

    refusal_agent = 1 if "insufficient" in agent_answer.lower() else 0

    results.append({
        "question": question,
        "sim_baseline": sim_baseline,
        "sim_agent": sim_agent,
        "hallucination_baseline": hallucination_baseline,
        "hallucination_agent": hallucination_agent,
        "len_baseline": len(baseline_answer.split()),
        "len_agent": len(agent_answer.split()),
        "refusal_agent": refusal_agent,
        "baseline_time": baseline_time,
        "agent_total_time": agent_total_time,
        "agent_retrieval_time": state.retrieval_time or 0,
        "agent_generation_time": state.generation_time or 0
    })
    
    print(f"  Baseline: {baseline_time:.2f}s | Agent: {agent_total_time:.2f}s")

# ---------- OOD ----------
print("\n===== OOD EVALUATION =====")

ood_results = []

for q in OOD_QUERIES:

    try:
        baseline_answer = baseline_rag(q) or ""
    except:
        baseline_answer = ""

    state = State(query=q)
    memory = run_agent(state)
    agent_answer = memory[0]["text"] if memory else ""

    baseline_h = is_hallucinated(baseline_answer, [])
    agent_h = is_hallucinated(agent_answer, [])

    ood_results.append((baseline_h, agent_h))

baseline_ood = sum(x[0] for x in ood_results) / len(ood_results)
agent_ood = sum(x[1] for x in ood_results) / len(ood_results)

print("Baseline OOD:", baseline_ood)
print("Agent OOD:", agent_ood)

# ---------- SAVE ----------
df = pd.DataFrame(results)

save_path = os.path.join(BASE_DIR, "evaluation", "results.csv")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)

print("\n===== FINAL METRICS =====")

print("\n📊 Similarity Scores:")
print(f"  Baseline: {df['sim_baseline'].mean():.4f} (±{df['sim_baseline'].std():.4f})")
print(f"  Agent: {df['sim_agent'].mean():.4f} (±{df['sim_agent'].std():.4f})")
print(f"  Improvement: {(df['sim_agent'].mean() - df['sim_baseline'].mean()):.4f}")

print("\n🚨 Hallucination Rates:")
print(f"  Baseline: {df['hallucination_baseline'].mean():.4f}")
print(f"  Agent: {df['hallucination_agent'].mean():.4f}")
print(f"  Reduction: {(df['hallucination_baseline'].mean() - df['hallucination_agent'].mean()):.4f}")

print("\n⏱️ Execution Time:")
print(f"  Baseline avg: {df['baseline_time'].mean():.2f}s")
print(f"  Agent avg total: {df['agent_total_time'].mean():.2f}s")
print(f"    - Retrieval: {df['agent_retrieval_time'].mean():.2f}s")
print(f"    - Generation: {df['agent_generation_time'].mean():.2f}s")

print("\n📝 Response Length:")
print(f"  Baseline avg: {df['len_baseline'].mean():.0f} words")
print(f"  Agent avg: {df['len_agent'].mean():.0f} words")

print(f"\n🙅 Refusal Rate (Agent): {df['refusal_agent'].mean():.4f}")
print(f"🌍 OOD Hallucination - Baseline: {baseline_ood:.4f}, Agent: {agent_ood:.4f}")


