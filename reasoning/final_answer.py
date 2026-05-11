from utils.llm import call_llm


def generate_final_answer(query, memory):

    context = " ".join([m["text"] for m in memory])
    prompt = f"""
You are an expert assistant answering questions based ONLY on verified knowledge.

Knowledge:
{context}

Question:
{query}

Instructions:
- Give a clear and detailed explanation
- Structure the answer logically
- Include:
    1. Definition (if applicable)
    2. Explanation
    3. Key insights or important details
- Do NOT hallucinate beyond the provided knowledge
- Do NOT be overly short

Answer:
"""

    return call_llm(prompt)