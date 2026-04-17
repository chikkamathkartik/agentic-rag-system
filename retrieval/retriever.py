from retrieval.faiss_store import FAISSStore
from retrieval.loader import load_text_file, chunk_text
from retrieval.store import faiss_store
DOCUMENTS = [
    "Deadlock is a condition where processes wait indefinitely for resources.",
    "Deadlock occurs due to mutual exclusion, hold and wait, no preemption, circular wait.",
    "Deadlock prevention avoids at least one necessary condition.",
    "Deadlock avoidance uses Banker’s algorithm.",
    "Deadlock detection finds cycles in resource allocation graph.",
    "Deadlock recovery involves terminating processes.",
]

# initialize FAISS once
store = FAISSStore()


def initialize_retriever(file_path):
    text = load_text_file(file_path)
    chunks = chunk_text(text)

    print(f"Loaded {len(chunks)} chunks")

    store.build_index(chunks)


def retrieve(query):
    return faiss_store.search(query)