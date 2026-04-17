import os

def load_documents(folder="uploads"):
    docs = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())

    return docs