from retrieval.store import faiss_store


def retrieve(query, top_k=5):

    enhanced_query = query + " causes risk factors reasons"
    results = faiss_store.search(enhanced_query, top_k=top_k)

    if not results:
        return ""

    return " ".join(results)