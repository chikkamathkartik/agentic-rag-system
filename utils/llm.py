import requests
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text)

def call_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        data = response.json()

        #  handle failure properly
        if "response" not in data:
            print("LLM Raw Error:", data)
            return None

        return data["response"].strip()

    except Exception as e:
        print("LLM Exception:", e)
        return None