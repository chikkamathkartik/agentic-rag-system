from utils.llm import call_llm


def evaluate_retrieval(query, context):
    if not context.strip():
        return "BAD"

    prompt = f"""
You are a strict retrieval evaluator.

Query:
{query}

Context:
{context}

Evaluate:

1. Does the context contain a clear definition?
2. Does it explain causes or conditions?
3. Does it include prevention or explanation?

Scoring Rules:
- If most key aspects are present → GOOD
- If only partial info → PARTIAL
- If irrelevant or missing → BAD

Answer ONLY one word:
GOOD / PARTIAL / BAD
"""

    response = call_llm(prompt)
    if not response:
        return "BAD"

    verdict = response.strip().upper().rstrip(".!")
    return verdict if verdict in {"GOOD", "PARTIAL", "BAD"} else "BAD"