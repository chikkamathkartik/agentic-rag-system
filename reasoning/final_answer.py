from utils.llm import call_llm


def generate_final_answer(query, memory):

    context = " ".join([m["text"] for m in memory])

    prompt = f"""
You are an expert assistant.

Use the verified knowledge below to answer:

{context}

Question:
{query}

Instructions:
- Give a clear and detailed explanation
- Include definition, explanation, and context
- Do NOT be overly short
- Do NOT hallucinate beyond given knowledge

Answer:
"""

    return call_llm(prompt)