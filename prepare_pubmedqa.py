import json
import os

INPUT_PATH = "data/ori_pqal.json"
DOCS_PATH = "dataset/docs"
QUERIES_PATH = "dataset/queries.json"

os.makedirs(DOCS_PATH, exist_ok=True)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

queries = []

for i, (pmid, item) in enumerate(data.items()):

    # -------- DOCUMENT --------
    context_text = " ".join(item["CONTEXTS"])

    with open(f"{DOCS_PATH}/doc_{i}.txt", "w", encoding="utf-8") as f:
        f.write(context_text)

    # -------- QUERY --------
    queries.append({
        "id": pmid,
        "question": item["QUESTION"],
        "ground_truth": item["LONG_ANSWER"]
    })

# Save queries
with open(QUERIES_PATH, "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2)

print("Dataset prepared ✔")