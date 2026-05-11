from utils.llm import call_llm


def generate(state):

    memory_context = " ".join([m["text"] for m in state.memory])

    prompt = f"""
Answer the question using the context.

Question:
{state.query}

Context:
{state.context}

Memory:
{memory_context}

Rules:
- Do NOT infer or assume anything
- Do NOT use outside knowledge
- ONLY use information explicitly present in the context
- If the answer is not directly present → say "Insufficient information"

Answer:
"""

    return call_llm(prompt)