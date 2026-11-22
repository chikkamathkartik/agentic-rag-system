# Manuscript Revision and Experiment Protocol

This repository contains the complete implementation, experimental evaluation suite, formal revision documents, and point-by-point reviewer rebuttal addressing the reviewer critique.

## Document Directory
- **Full Revised Manuscript**: [REVISED_MANUSCRIPT.md](file:///e:/DOC/REVISED_MANUSCRIPT.md)
  - Contains all 14 publication sections: Introduction, Related Work, Formal Problem Formulation, Algorithmic Details, Baseline Configurations, Results, Ablations, Robustness, Faithfulness Metrics, Latency Profile, Human Evaluation, Five-Class Failure Taxonomy, and Reproducibility Specifications.
- **Formal Point-by-Point Rebuttal**: [RESPONSE_TO_REVIEWERS.md](file:///e:/DOC/RESPONSE_TO_REVIEWERS.md)
  - Answers every single sentence of the reviewer's feedback with references to sections, tables, code modules, and tests.
- **Evaluation Guide**: [EVALUATION_GUIDE.md](file:///e:/DOC/EVALUATION_GUIDE.md)

---

## 1. Research Gap and Novelty Claim
The contribution is rigorously grounded and positioned against contemporary selective QA and agentic RAG literature:
- We do not claim that "agentic RAG" or "refusal" is novel.
- Instead, we address the precise research gap: existing RAG systems either treat retrieval and generation as independent decoupled stages, or deploy heavy, multi-turn reflective agents (Self-RAG, CRAG) requiring customized model fine-tuning.
- We formally introduce and empirically validate a lightweight, model-agnostic, dual-signal gating policy combining continuous dense retrieval similarity ($r$) with discrete, categorical evidence coverage judgments ($g$).

---

## 2. Formal Mathematical Definition of the Trust-Aware Mechanism
For query $q$ and retrieved passages $D_k$:
1. $s_i = \cos(\phi(q), \phi(d_{(i)})) \in [-1, 1]$
2. $r(q, D_k) = \text{clip}(\max_{i} s_i, 0, 1)$
3. $v = \mathcal{M}_{\text{critic}}(q, D_k) \in \{\text{BAD}, \text{PARTIAL}, \text{GOOD}\}$
4. $g(v) = 0.0 \text{ (BAD)}, 0.5 \text{ (PARTIAL)}, 1.0 \text{ (GOOD)}$
5. $T(q, D_k, v) = \lambda \cdot r(q, D_k) + (1 - \lambda) \cdot g(v)$
6. Decision rule:
   $$\delta(q, D_k, v) = \begin{cases} \text{ABSTAIN}, & \text{if } v = \text{BAD} \;\;\lor\;\; T(q, D_k, v) < \tau \\ \text{ANSWER}, & \text{otherwise} \end{cases}$$
Defaults: $\lambda = 0.50$, $\tau = 0.50$. $T$ is an ordinal decision index, not a calibrated probability.

---

## 3. Implemented Baselines, Ablations, and Tools

### Evaluated Systems (`evaluation/run_revision_eval.py`)
1. **Dense RAG**: `all-MiniLM-L6-v2` FAISS `IndexFlatIP`.
2. **Sparse BM25 RAG**: Okapi BM25 ($k_1=1.5, b=0.75$).
3. **Hybrid RRF RAG**: Reciprocal Rank Fusion ($c=60$) combining dense and BM25 rankings.
4. **Cross-Encoder Rerank RAG**: Two-stage dense candidate selection + cross-attentive token scoring.
5. **Relevance-Only Gated RAG**: Dense retrieval gated by $r \ge \tau$.
6. **Critic-Only Gated RAG**: Dense retrieval gated by $v \ne \text{BAD}$.
7. **Trust-Aware Dense RAG**: Full combined policy ($T \ge \tau \land v \ne \text{BAD}$).

### Evaluation & Analysis Scripts
- **Main Evaluation Runner**: `evaluation/run_revision_eval.py`
- **Robustness & OOD Stress Suite**: `evaluation/robustness_eval.py` (Tests typo noise, distractor injection, and out-of-domain unanswerable queries)
- **Ablation & Sensitivity Grid**: `evaluation/ablation_study.py` (Sweeps $\lambda \in [0, 1]$ and $\tau \in [0, 1]$)
- **Automated Failure Mode Classifier**: `evaluation/failure_analysis.py` (Categorizes failures across Types I–V)
- **Blinded Human Annotation Generator**: `evaluation/create_human_eval.py`
- **Verified Unit Tests**: `tests/test_trust_policy.py` (13 tests verifying policy, baselines, metrics, noise, failure categorization, and blinding)

---

## 4. Execution Commands

### Run Unit Test Suite
```powershell
python -m unittest discover tests
```

### Run Main Comparison (Smoke Test or Full 1,000 Dataset)
```powershell
# Quick smoke test
python evaluation\run_revision_eval.py --limit 30 --seed 13

# Full benchmark (1,000 queries)
python evaluation\run_revision_eval.py --limit 1000 --seed 13
```

### Run Robustness & OOD Stress Test
```powershell
python evaluation\robustness_eval.py --limit 15 --seed 13
```

### Run Ablation Grid & Failure Analysis
```powershell
python evaluation\ablation_study.py
python evaluation\failure_analysis.py
```

### Generate Blinded Human Evaluation Sheet
```powershell
python evaluation\create_human_eval.py --seed 13
```
