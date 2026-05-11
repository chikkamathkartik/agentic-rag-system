from retrieval.faiss_store import FAISSStore
import os
from utils.chunker import chunk_text

# ✅ GLOBAL INSTANCE
faiss_store = FAISSStore()

# ✅ Prevent duplicate loading
_data_loaded = False


def load_dataset_into_faiss(path="dataset/docs"):

    global faiss_store, _data_loaded

    # 🚨 Avoid reloading same data
    if _data_loaded:
        print("[FAISS] Dataset already loaded. Skipping...")
        return

    if not os.path.exists(path):
        print(f"[FAISS] ❌ Path not found: {path}")
        return

    all_chunks = []
    file_count = 0

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if not filename.endswith(".txt"):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        file_count += 1

    clean_chunks = [c for c in all_chunks if len(c.strip()) > 30]

    if not clean_chunks:
        print("[FAISS] ❌ No valid chunks found")
        return

    faiss_store.add(clean_chunks)

    _data_loaded = True

    print(f"[FAISS] Loaded {file_count} files")
    print(f"[FAISS] Total chunks: {len(clean_chunks)}")