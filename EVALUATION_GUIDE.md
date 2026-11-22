# Evaluation Guide

This guide provides instructions for replicating the benchmark experiments, baseline comparisons, ablation studies, robustness tests, and human evaluation protocols detailed in [REVISED_MANUSCRIPT.md](file:///e:/DOC/REVISED_MANUSCRIPT.md) and [RESPONSE_TO_REVIEWERS.md](file:///e:/DOC/RESPONSE_TO_REVIEWERS.md).

---

## 1. Unit Tests

Before running benchmarks, verify system components:
```powershell
python -m unittest discover tests
```
Runs 13 automated tests covering:
- Trust policy mathematical boundedness and fail-closed safety
- Okapi BM25 and Cross-Encoder reranking baselines
- Claim-level faithfulness and hallucination metrics
- Selective prediction metrics (coverage, risk, selective accuracy)
- Bootstrap confidence interval calculations
- Typo perturbation noise generator
- Failure mode categorization logic
- Double-blind human evaluation pairing and key separation

---

## 2. Comparative Evaluation Across RAG Baselines

The main runner (`evaluation/run_revision_eval.py`) benchmarks:
1. `dense_rag`: Dense retrieval with `all-MiniLM-L6-v2` FAISS `IndexFlatIP`.
2. `bm25_rag`: Full corpus Okapi BM25 ranking.
3. `hybrid_rag`: Reciprocal Rank Fusion (RRF, $c=60$) combining dense and sparse ranks.
4. `cross_encoder_rag`: Two-stage candidate retrieval with cross-attention / token-interaction scoring.
5. `relevance_only_dense`: Ablation gated purely by cosine similarity ($r \ge 0.5$).
6. `critic_only_dense`: Ablation gated purely by LLM critic verdict ($v \ne \text{BAD}$).
7. `trust_aware_dense`: Combined dual-signal trust gating ($T \ge 0.5 \land v \ne \text{BAD}$).

### Commands
```powershell
# Smoke test (30 sampled queries)
python evaluation\run_revision_eval.py --limit 30 --seed 13

# Full benchmark (All 1,000 PubMedQA queries)
python evaluation\run_revision_eval.py --limit 1000 --seed 13

# Plot results
python evaluation\plot.py
```
Output is saved to `evaluation/revision_results.csv` and plots to `evaluation/revision_plots/`.

---

## 3. Robustness & Out-of-Domain (OOD) Stress Testing

Evaluates resilience against query typos, adversarial distractor passage injection, and out-of-domain unanswerable queries:
```powershell
python evaluation\robustness_eval.py --limit 15 --seed 13
```
Output is saved to `evaluation/robustness_results.csv`.

---

## 4. Hyperparameter Sensitivity & Ablation Grid

Runs a parameter sensitivity grid across mixing weight $\lambda \in [0.0, 1.0]$ and threshold $\tau \in [0.20, 0.80]$:
```powershell
python evaluation\ablation_study.py --input evaluation\revision_results.csv --output evaluation\ablation_grid_results.csv
```

---

## 5. Automated Failure Mode Analysis

Categorizes evaluation rows into the five-class failure taxonomy:
- Type I: Retrieval Miss
- Type II: Incomplete Evidence Abstention
- Type III: Critic False Negative / Positive
- Type IV: Generator Ungrounded Hallucination
- Type V: Over-cautious Abstention

```powershell
python evaluation\failure_analysis.py --input evaluation\revision_results.csv --output evaluation\failure_analysis_report.md
```

---

## 6. Double-Blind Human Evaluation Protocol

Generates blinded pairwise comparison sheets where system identities are stripped and replaced with randomized `A` / `B` labels:
```powershell
python evaluation\create_human_eval.py --input evaluation\revision_results.csv --output evaluation\human_eval.csv --seed 13
```
- Distribute `evaluation/human_eval.csv` to annotators.
- Keep `evaluation/human_eval_key.csv` securely segregated until adjudication.
