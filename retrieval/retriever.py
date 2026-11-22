from retrieval.store import faiss_store


def retrieve_scored(query, top_k=5):
    enhanced_query = query + " causes risk factors reasons"
    return faiss_store.search_with_scores(enhanced_query, top_k=top_k)


def retrieve(query, top_k=5):
    return [text for text, _ in retrieve_scored(query, top_k=top_k)]