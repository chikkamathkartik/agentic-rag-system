from flask import Flask, render_template, request, redirect, session
import os

from baseline import baseline_rag
from core.agent import run_agent
from core.state import State

from reasoning.final_answer import generate_final_answer
from utils.chunker import chunk_text
from utils.file_loader import load_documents
from retrieval.store import faiss_store


app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.secret_key = "secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- FORMAT OUTPUT ----------------
def format_output(memory):
    sections = {
        "definition": [],
        "cause": [],
        "condition": [],
        "prevention": [],
        "example": [],
        "other": []
    }

    def classify(text):
        t = text.lower()

        if any(x in t for x in ["is defined as", "refers to", "is a", "is an"]):
            return "definition"

        if any(x in t for x in ["cause", "occurs", "happens due to", "leads to"]):
            return "cause"

        if any(x in t for x in ["mutual exclusion", "hold and wait", "circular wait"]):
            return "condition"

        if any(x in t for x in ["prevent", "avoid", "eliminate"]):
            return "prevention"

        if "example" in t:
            return "example"

        return "other"

    for m in memory:
        sections[classify(m["text"])].append(m["text"])

    return sections


# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    # ---------- INIT SESSION ----------
    if "history" not in session:
        session["history"] = []

    # ---------- DEFAULT VALUES ----------
    baseline_output = None
    agent_output = None
    final_answer = None

    uploaded_files = os.listdir(app.config["UPLOAD_FOLDER"])

    # ---------------- POST ----------------
    if request.method == "POST":

        # -------- FILE UPLOAD --------
        files = request.files.getlist("files")

        if files and any(f.filename for f in files):

            new_chunks = []

            for file in files:
                if file.filename:
                    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                    file.save(path)

            # Load and process ALL docs
            docs = load_documents()

            for doc in docs:
                new_chunks.extend(chunk_text(doc))

            # Filter bad chunks
            clean_chunks = [c for c in new_chunks if len(c.strip()) > 30]

            if clean_chunks:
                faiss_store.add(clean_chunks)

        # -------- QUERY --------
        query = request.form.get("query")

        if query:

            # Agent
            state = State(query=query)
            memory = run_agent(state)

            agent_output = format_output(memory)

            # Final explanation
            final_answer = generate_final_answer(query, memory)

            # Save history
            session["history"].append({
                "query": query,
                "answer": final_answer
            })

    # ---------------- RENDER ----------------
    return render_template(
        "index.html",
        baseline_output=baseline_output,
        agent_output=agent_output,
        final_answer=final_answer,
        files=uploaded_files,
        history=session["history"]
    )


# ---------------- CLEAR ROUTE ----------------
@app.route("/clear")
def clear():
    session.clear()
    return redirect("/")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)