from retrieval.retriever import retrieve
from retrieval.critic import evaluate_retrieval
from reasoning.generator import generate
import time


def run_agent(state):

    print("\n[Step 0] RETRIEVE")
    
    retrieval_start = time.time()
    context = retrieve(state.query)
    state.retrieval_time = time.time() - retrieval_start

    verdict = evaluate_retrieval(state.query, context)
    print("Retrieval Critic:", verdict)

    # 🚨 HARD STOP (NO HALLUCINATION)
    if verdict == "BAD":
        print("❌ No relevant context → stopping")
        state.end_time = time.time()
        return []

    # ---------------- REASON ----------------
    print("\n[Step 1] REASON")

    state.context = context
    
    generation_start = time.time()
    answer = generate(state)
    state.generation_time = time.time() - generation_start

    print("FINAL ANSWER:", answer)
    
    # Finalize execution time
    state.end_time = time.time()

    return [{
        "text": answer,
        "confidence": 1.0
    }]