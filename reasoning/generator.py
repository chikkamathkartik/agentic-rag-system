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
- Give clear factual answer
- Avoid repeating memory
- Be concise (2–4 sentences)
"""

    return call_llm(prompt)