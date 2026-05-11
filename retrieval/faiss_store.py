import faiss
import numpy as np
from utils.llm import get_embedding


class FAISSStore:

    def __init__(self, dim=384):
        self.index = faiss.IndexFlatIP(dim)   # ✅ better than L2
        self.texts = []

    def add(self, chunks):
        embeddings = [get_embedding(c) for c in chunks]
        vectors = np.array(embeddings).astype("float32")

        faiss.normalize_L2(vectors)  # ✅ important

        self.index.add(vectors)
        self.texts.extend(chunks)

    def search(self, query, top_k=3):

        if len(self.texts) == 0:
            return []

        top_k = min(top_k, len(self.texts)) 

        q_vec = np.array([get_embedding(query)]).astype("float32")
        faiss.normalize_L2(q_vec) 

        distances, indices = self.index.search(q_vec, top_k)

        results = []

        for i in indices[0]:
            if 0 <= i < len(self.texts):
                results.append(self.texts[i])

        return results