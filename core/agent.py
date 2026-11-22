from retrieval.retriever import retrieve_scored
from retrieval.critic import evaluate_retrieval
from reasoning.generator import generate
from core.trust_policy import assess_retrieval
import time


def run_agent(state):

    print("\n[Step 0] RETRIEVE")

    retrieval_start = time.perf_counter()
    scored_context = retrieve_scored(state.query)
    state.retrieval_time = time.perf_counter() - retrieval_start
    context_chunks = [text for text, _ in scored_context]
    context = "\n".join(context_chunks)

    critic_start = time.perf_counter()
    verdict = evaluate_retrieval(state.query, context)
    state.critic_time = time.perf_counter() - critic_start
    assessment = assess_retrieval(
        [score for _, score in scored_context],
        verdict,
    )
    state.retrieval_verdict = assessment.verdict
    state.retrieval_relevance = assessment.retrieval_relevance
    state.trust_score = assessment.trust_score
    state.context = context
    print("Retrieval Critic:", verdict)

    if assessment.abstain:
        state.abstained = True
        print("Insufficient trusted evidence; stopping.")
        state.end_time = time.time()
        return []

    print("\n[Step 1] REASON")

    generation_start = time.perf_counter()
    answer = generate(state)
    state.generation_time = time.perf_counter() - generation_start
    if not answer:
        raise RuntimeError("The configured LLM returned no answer.")

    print("FINAL ANSWER:", answer)
    state.end_time = time.time()

    return [{
        "text": answer,
        "confidence": state.trust_score
    }]