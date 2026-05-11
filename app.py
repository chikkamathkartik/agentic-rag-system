from flask import Flask, render_template, request, redirect
import os
import shutil

from core.agent import run_agent
from core.state import State
from utils.chunker import chunk_text
from utils.file_loader import load_documents
from retrieval.store import faiss_store
from retrieval.store import faiss_store, load_dataset_into_faiss

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    final_answer = None
    execution_time = None
    uploaded_files = os.listdir(app.config["UPLOAD_FOLDER"])

    if request.method == "POST":

        # -------- FILE UPLOAD --------
        files = request.files.getlist("files")

        if files and any(f.filename for f in files):

            new_chunks = []

            for file in files:
                if file.filename:
                    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                    file.save(path)

            docs = load_documents()

            for doc in docs:
                new_chunks.extend(chunk_text(doc))

            clean_chunks = [c for c in new_chunks if len(c.strip()) > 30]

            if clean_chunks:
                faiss_store.add(clean_chunks)

        # -------- QUERY --------
        query = request.form.get("query")

        if query:

            print("QUERY RECEIVED:", query)

            state = State(query=query)
            memory = run_agent(state)
            
            # Get execution time
            execution_time = state.get_execution_time()

            print("MEMORY:", memory)

            if not memory:
                final_answer = "No documents available or no relevant information found. Please upload documents."
            
            else:
                final_answer = memory[0]["text"]

            print("FINAL ANSWER:", final_answer)
            print(f"EXECUTION TIME: {execution_time:.2f}s")

    return render_template(
        "index.html",
        final_answer=final_answer,
        execution_time=execution_time,
        files=uploaded_files
    )


# ---------------- DELETE SINGLE FILE ----------------
@app.route("/delete/<filename>")
def delete_file(filename):

    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if os.path.exists(path):
        os.remove(path)

    # ⚠️ Reset FAISS to avoid stale data
    faiss_store.index.reset()
    faiss_store.texts = []

    return redirect("/")


# ---------------- CLEAR ALL DOCUMENTS ----------------
@app.route("/clear_docs")
def clear_docs():

    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Reset FAISS completely
    faiss_store.index.reset()
    faiss_store.texts = []

    return redirect("/")


# ---------------- RUN ----------------
if __name__ == "__main__":
    load_dataset_into_faiss()
    app.run(host="0.0.0.0", port=5000, debug=True)