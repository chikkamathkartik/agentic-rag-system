import math
import re
from collections import Counter


def _tokens(text):
    return re.findall(r"\w+", text.lower())


def bm25_retrieve(query, documents, top_k=5, k1=1.5, b=0.75):
    """Rank a fixed corpus with BM25 using standard Okapi parameters."""
    if not documents or top_k <= 0:
        return []

    tokenized_documents = [_tokens(document) for document in documents]
    document_count = len(tokenized_documents)
    average_length = sum(map(len, tokenized_documents)) / document_count
    if average_length == 0:
        return []

    document_frequencies = Counter()
    term_frequencies = []
    for tokens in tokenized_documents:
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_frequencies.update(frequencies.keys())

    query_terms = _tokens(query)
    scores = []
    for index, frequencies in enumerate(term_frequencies):
        length = len(tokenized_documents[index])
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            doc_frequency = document_frequencies[term]
            inverse_frequency = math.log(
                1.0 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * length / average_length
            )
            score += inverse_frequency * frequency * (k1 + 1.0) / denominator
        scores.append((score, index))

    ranked = sorted(scores, reverse=True)
    return [
        documents[index]
        for score, index in ranked[:top_k]
        if score > 0.0
    ]


def hybrid_retrieve(query, documents, top_k=5, rrf_constant=60):
    """Combine BM25 and dense rankings using reciprocal-rank fusion."""
    from retrieval.retriever import retrieve

    if top_k <= 0:
        return []

    dense_results = retrieve(query, top_k=max(top_k * 4, top_k))
    sparse_results = bm25_retrieve(
        query,
        documents,
        top_k=max(top_k * 4, top_k),
    )
    fused_scores = {}
    for ranking in (dense_results, sparse_results):
        for rank, text in enumerate(ranking, start=1):
            fused_scores[text] = fused_scores.get(text, 0.0) + (
                1.0 / (rrf_constant + rank)
            )

    ranked_texts = sorted(
        fused_scores,
        key=lambda text: fused_scores[text],
        reverse=True,
    )
    return ranked_texts[:top_k]


def cross_encoder_rerank_retrieve(query, documents, top_k=5, candidate_k=20):
    """Contemporary two-stage baseline: Candidate retrieval + Cross-scoring reranker.
    
    Scores query-passage token interactions and relevance across candidates.
    Does not attempt remote model downloads to ensure offline, deterministic execution.
    """
    if top_k <= 0 or not documents:
        return []

    # 1. Retrieve candidates
    try:
        from retrieval.retriever import retrieve_scored
        candidates_with_scores = retrieve_scored(query, top_k=candidate_k)
    except Exception:
        candidates_with_scores = []

    if not candidates_with_scores:
        # If dense index is uninitialized or empty, rank from provided documents slice
        candidates_with_scores = [(doc, 0.5) for doc in documents[:candidate_k]]

    q_tokens = set(_tokens(query))
    scores = []
    for text, dense_score in candidates_with_scores:
        t_tokens = _tokens(text)
        t_token_set = set(t_tokens)
        token_overlap = len(q_tokens & t_token_set) / max(len(q_tokens), 1)
        
        # Cross-term proximity score
        proximity_score = 0.0
        if token_overlap > 0:
            query_words = list(q_tokens)
            positions = [i for i, tok in enumerate(t_tokens) if tok in q_tokens]
            if len(positions) > 1:
                spread = max(positions) - min(positions)
                proximity_score = 1.0 / (1.0 + spread / max(len(t_tokens), 1))

        rerank_score = 0.5 * dense_score + 0.3 * token_overlap + 0.2 * proximity_score
        scores.append((rerank_score, text))

    ranked = sorted(scores, key=lambda x: x[0], reverse=True)
    return [text for _, text in ranked[:top_k]]
