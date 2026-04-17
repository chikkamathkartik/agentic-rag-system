from utils.llm import call_llm


def evaluate_retrieval(query, context):

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

    try:
        response = call_llm(prompt)

        if not response:
            return "BAD"

        response = response.strip().upper()

        if "GOOD" in response:
            return "GOOD"
        elif "PARTIAL" in response:
            return "PARTIAL"
        else:
            return "BAD"

    except:
        return "BAD"