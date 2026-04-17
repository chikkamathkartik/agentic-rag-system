import faiss
import numpy as np
from utils.llm import get_embedding


class FAISSStore:

    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, chunks):
        embeddings = [get_embedding(c) for c in chunks]
        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)
        self.texts.extend(chunks)

    # ✅ THIS METHOD MUST EXIST
    def search(self, query, k=3):

        if len(self.texts) == 0:
            return []

        q_vec = np.array([get_embedding(query)]).astype("float32")

        distances, indices = self.index.search(q_vec, k)

        results = []

        for i in indices[0]:
            if 0 <= i < len(self.texts):
                results.append(self.texts[i])

        return results