import os
import requests

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# Backwards compatibility for existing imports
class _LazyModelProxy:
    def encode(self, *args, **kwargs):
        return get_model().encode(*args, **kwargs)


model = _LazyModelProxy()


def get_embedding(text):
    return get_model().encode(text)


def call_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
            },
            timeout=30,
        )

        data = response.json()

        if "response" not in data:
            print("LLM Raw Error:", data)
            return None

        return data["response"].strip()

    except Exception as e:
        print("LLM Exception:", e)
        return None
