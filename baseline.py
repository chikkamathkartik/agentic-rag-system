from retrieval.retriever import retrieve
from utils.llm import call_llm


def baseline_rag(query):
    docs = retrieve(query)
    context = "\n".join(docs)

    prompt = f"""
Answer the question using the context.

Question: {query}

Context:
{context}

Give a clear answer.
"""

    return call_llm(prompt)