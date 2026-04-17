# Controlled Agentic RAG System

This project implements a controlled Agentic Retrieval-Augmented Generation (RAG) pipeline with:

- Retrieval Critic (validates retrieved context)
- Step-level verification (checks reasoning quality)
- Confidence-based memory (stores only useful knowledge)
- Budget-aware execution (avoids unnecessary computation)
- Loop control (prevents redundant reasoning)

## Features

- Upload documents (PDF, TXT, DOCX)
- Chunking + FAISS-based retrieval
- Agentic reasoning with verification
- Final answer generation
- Clean Flask-based UI

## Architecture

Upload → Chunk → FAISS → Retrieve → Reason → Verify → Memory → Final Answer

## How to Run

```bash
pip install -r requirements.txt
python app.py