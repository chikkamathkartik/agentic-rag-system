# Quick Start

## Setup

Install dependencies and ensure Ollama is running with the configured `llama3` model:

```powershell
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000` to use the web interface.

## Evaluate

The evaluation runner compares dense, BM25, hybrid reciprocal-rank-fusion, and trust-aware dense RAG. It initializes the document index and leaves the existing `evaluation/results.csv` untouched.

```powershell
python evaluation\run_revision_eval.py --limit 30 --seed 13
python evaluation\plot.py
```

For all 1,000 dataset entries:

```powershell
python evaluation\run_revision_eval.py --limit 1000 --seed 13
```

Results and figures are written to `evaluation/revision_results.csv` and `evaluation/revision_plots/`. Use `evaluation\run_evaluation.py` as a compatibility alias.

Create a blinded human comparison sheet from the evaluation output:

```powershell
python evaluation\create_human_eval.py --seed 13
```

The annotation sheet is `evaluation/human_eval.csv`; the unblinding key is kept separately in `evaluation/human_eval_key.csv`. The script only prepares the form; people still need to complete and analyze the study.

## Important limitations

The repository does not yet include a completed human study, validated faithfulness/hallucination metric, OOD benchmark, robustness suite, bootstrap analysis, or complete token/hardware cost tracking. Similarity scores are proxies, not factual correctness. Read [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) and [MANUSCRIPT_REVISION.md](./MANUSCRIPT_REVISION.md) before making paper claims.
