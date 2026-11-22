# Calibrated Evidence Gating for Selective Retrieval-Augmented Generation: A Trust-Aware Answer/Abstain Mechanism

**Anonymous Authors**  
*Under Review for Publication*

---

## Abstract

Retrieval-Augmented Generation (RAG) significantly enhances the factual grounding of large language models (LLMs) by conditioning generation on retrieved external documents. However, contemporary RAG systems remain vulnerable to retrieval failures, partial evidence coverage, and silent hallucinations when presented with ambiguous, incomplete, or out-of-domain (OOD) queries. In this work, we present a formal study and empirical evaluation of a lightweight, trust-aware answer/abstain decision mechanism for selective RAG. Rather than claiming a novel architectural paradigm, we address a precise research gap: whereas existing RAG frameworks evaluate retrieval quality and generation quality independently or rely on monolithic end-to-end verifiers, our mechanism integrates dense continuous retrieval similarity with a categorical LLM-based evidence coverage assessment into a scalar decision index $T(q, D_k, v) \in [0, 1]$. We establish formal bounding properties for the decision rule, prove fail-closed guarantees under retrieval ambiguity, and provide algorithmic details. We conduct comprehensive empirical benchmarking against contemporary retrieval baselines—including Dense RAG (`all-MiniLM-L6-v2`), Okapi BM25, Hybrid Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking—under a shared generator (`llama3`) on the biomedical PubMedQA corpus. We report extensive component ablations, hyperparameter sensitivity sweeps ($\lambda, \tau$), character noise perturbations, distractor injections, OOD abstention stress testing, claim-level faithfulness and hallucination rates, stage-by-stage computational latency, and a double-blinded human evaluation protocol with inter-annotator agreement metrics. Finally, we provide an empirical five-class failure taxonomy with representative case studies and complete reproducibility specifications.

---

## 1. Introduction & Clarified Novelty Claims

### 1.1 The Challenge of Trustworthy Selective RAG
Retrieval-Augmented Generation (RAG) pairs dense or sparse document retrieval with autoregressive language generation to overcome the static parameter memory limitations of LLMs. Despite wide adoption, standard RAG architectures operate under an *open-loop generation mandate*: regardless of whether the retrieved passages actually contain sufficient, faithful evidence to address the query, the generator is prompted to synthesize a coherent answer. When evidence is missing, contradictory, or marginal, this produces silent, fluent hallucinations that undermine deployment in high-stakes domains such as clinical medicine, law, and engineering.

### 1.2 Precise Research Gap
Prior literature has explored selective prediction (abstaining under uncertainty) and retrieval reflection:
1. **Selective Classification & QA**: Approaches such as softmax confidence thresholding or margin-based rejection typically rely purely on internal LLM log-probabilities, which are notoriously miscalibrated and unavailable behind black-box API endpoints.
2. **Reflective & Agentic RAG**: Systems such as Self-RAG (Asai et al., 2024), Corrective RAG (CRAG; Yan et al., 2024), and Adaptive RAG (Wang et al., 2024) introduce fine-tuned reflection tokens or multi-agent loops. However, these systems either require costly specialized model fine-tuning, execute heavy multi-turn conversational graphs, or decouple retrieval scoring from generation gating.
3. **The Unaddressed Gap**: There lacks a lightweight, mathematically bounded, model-agnostic decision policy that **jointly couples continuous vector similarity metrics ($r$) with discrete, categorical evidence coverage judgments ($g$)** to determine per-query answerability before committing expensive generation FLOPs.

### 1.3 Scope and Novelty Claims
To avoid over-claiming, we state our contributions conservatively:
- **Systematic Formalization**: We formally define a dual-signal trust-aware decision rule $T(q, D_k, v) = \lambda r(q, D_k) + (1-\lambda) g(v)$ that maps dense semantic alignment and categorical evidence coverage into a bounded decision index with fail-closed safety semantics.
- **Transparent Algorithmic Specifications**: We provide full algorithmic workflows, state transitions, prompt templates, and complexity bounds for both the gating mechanism and contemporary baselines (Dense, BM25, Hybrid RRF, and Cross-Encoder).
- **Rigorous Multi-Dimensional Evaluation**: We report comparative evaluations on PubMedQA across multiple contemporary baselines and ablations, replacing superficial cosine proxies with claim-level faithfulness metrics, selective risk-coverage curves, perturbation robustness, stage-by-stage latency profiling, and a blinded human annotation protocol.
- **Empirical Failure Taxonomy**: We categorize failure modes into a five-class taxonomy (Retrieval Miss, Incomplete Evidence, Critic Misjudgment, Generator Extrapolation, and Over-Cautious Abstention), offering qualitative case studies and actionable mitigations.

---

## 2. Related Work and Conceptual Positioning

Table 1 summarizes the conceptual positioning of our trust-aware policy relative to contemporary paradigms.

**Table 1: Conceptual Comparison of Retrieval-Augmented Generation Architectures**

| Paradigm | Mechanism | Abstention Capability | Requires Fine-tuning? | Computational Overhead |
| :--- | :--- | :---: | :---: | :---: |
| **Standard Dense RAG** (Lewis et al., 2020) | Top-$k$ dense retrieval + Generator | ❌ None (Forces generation) | ❌ No | Low ($1\times$ Retrieval, $1\times$ Gen) |
| **Hybrid BM25 + Dense RRF** (Cormack et al., 2009) | Reciprocal Rank Fusion of sparse + dense | ❌ None | ❌ No | Low ($2\times$ Retrieval, $1\times$ Gen) |
| **Self-RAG** (Asai et al., 2024) | Special reflection tokens (`[ISREL]`, `[ISSUP]`) | ✅ Yes | ✅ Yes (Custom 7B/13B LM) | Moderate ($1\times$ Gen with token search) |
| **Corrective RAG (CRAG)** (Yan et al., 2024) | Heuristic retrieval evaluator + Web search fallback | ⚠️ Fallback, rarely abstains | ⚠️ Yes (Lightweight evaluator) | High (Multi-stage web retrieval) |
| **Trust-Aware RAG (Ours)** | **Dual-signal ($r, g$) bounded trust gating** | **✅ Explicit Fail-Closed Abstention** | **❌ No (Off-the-shelf LLM & Embedder)** | **Low-Moderate ($1\times$ Retrieval, $1\times$ Critic, $\le 1\times$ Gen)** |

---

## 3. Formal Problem Formulation & Trust-Aware Mechanism

### 3.1 Mathematical Definitions
Let $\mathcal{C} = \{d_1, d_2, \dots, d_N\}$ denote a text corpus of document passages. Given an incoming natural language query $q \in \mathcal{Q}$:

1. **Dense Retrieval and Similarity Scores**:  
   A dual-encoder embedding function $\phi: \mathcal{X} \to \mathbb{R}^d$ maps text to a $d$-dimensional normalized vector space ($\|\phi(x)\|_2 = 1$). The dense index retrieves the top-$k$ passages $D_k = \{d_{(1)}, d_{(2)}, \dots, d_{(k)}\}$ maximizing the inner product:
   $$s_i = \langle \phi(q), \phi(d_{(i)}) \rangle = \cos(\phi(q), \phi(d_{(i)})), \quad \forall i \in \{1, \dots, k\}$$
   Since vectors are $\ell_2$-normalized, $s_i \in [-1, 1]$.

2. **Continuous Retrieval Relevance Measure $r(q, D_k)$**:  
   The relevance of the retrieved passage set is defined as the maximum similarity score clipped to the unit interval:
   $$r(q, D_k) = \text{clip}\left(\max_{i \in \{1, \dots, k\}} s_i, \, 0, \, 1\right) \in [0, 1]$$

3. **Categorical Critic Verdict $v$ and Evidence Coverage Mapping $g(v)$**:  
   An LLM-based retrieval critic $\mathcal{M}_{\text{critic}}$ assesses whether the concatenated retrieved passages $D_k$ provide sufficient factual evidence to answer query $q$:
   $$v = \mathcal{M}_{\text{critic}}(q, D_k) \in \{\text{BAD}, \, \text{PARTIAL}, \, \text{GOOD}\}$$
   The categorical verdict is mapped to a discrete numerical evidence score via monotonic mapping function $g: \mathcal{V} \to [0, 1]$:
   $$g(v) = \begin{cases} 0.0, & \text{if } v = \text{BAD} \\ 0.5, & \text{if } v = \text{PARTIAL} \\ 1.0, & \text{if } v = \text{GOOD} \end{cases}$$
   If $\mathcal{M}_{\text{critic}}$ outputs an invalid or unparseable token, the system enforces a fail-closed default: $v \leftarrow \text{BAD} \implies g(v) = 0.0$.

4. **Fused Trust Decision Index $T(q, D_k, v)$**:  
   The fused trust index combines continuous embedding relevance and categorical evidence coverage using a mixing parameter $\lambda \in [0, 1]$:
   $$T(q, D_k, v) = \lambda \cdot r(q, D_k) + (1 - \lambda) \cdot g(v)$$
   
5. **Selective Decision Rule $\delta(q, D_k, v)$**:  
   Given a pre-specified decision threshold $\tau \in [0, 1]$, the system evaluates the selective action:
   $$\delta(q, D_k, v) = \begin{cases} \text{ABSTAIN}, & \text{if } v = \text{BAD} \;\;\lor\;\; T(q, D_k, v) < \tau \\ \text{ANSWER}, & \text{otherwise} \end{cases}$$

### 3.2 Formal Properties & Guarantees
- **Property 1 (Bounded Metric Space)**: Because $r(q, D_k) \in [0, 1]$ and $g(v) \in [0, 1]$, the trust score $T$ is convexly bounded:
  $$\forall \lambda \in [0, 1], \quad 0 \le T(q, D_k, v) \le 1$$
- **Property 2 (Fail-Closed Safety Dominance)**: Irrespective of dense cosine similarity, if the critic identifies that the passages lack factual coverage ($v = \text{BAD}$), the decision rule enforces $\delta = \text{ABSTAIN}$. Formally:
  $$\lim_{r \to 1.0} \delta(q, D_k, \text{BAD}) = \text{ABSTAIN}$$
  This prevents "topic-similarity hallucination," where passages share lexical overlap with the query but fail to answer the specific question asked.
- **Property 3 (Non-Calibrated Index Status)**: $T$ is an ordinal decision index optimized for selective gating; it is **not** a frequentist calibrated probability $\mathbb{P}(\text{Correct} \mid q)$.

---

## 4. Methodological and Algorithmic Details

### 4.1 System Architecture
The system operates as a pipelined agent with stateful tracking of timing, diagnostic metrics, and selective branches.

```mermaid
flowchart TD
    Q[User Query q] --> E[Query Expansion]
    E --> R[Dense Retrieval FAISS]
    R --> Dk[Top-k Passages D_k]
    Dk --> C[Retrieval Critic M_critic]
    Dk --> S[Max Cosine Sim r]
    C --> V[Critic Verdict v]
    V --> T[Trust Fusion T = lambda*r + 1-lambda*g]
    S --> T
    T --> Dec{v == BAD or T < tau?}
    Dec -- Yes --> Abs[ABSTAIN: Return Refusal / Empty]
    Dec -- No --> Gen[Generator M_gen: Synthesize Answer]
    Gen --> Out[Grounded Final Answer]
```

### 4.2 Algorithmic Pseudocode

```text
================================================================================
Algorithm 1: Trust-Aware Retrieval & Selective Abstention Policy
================================================================================
Input : Query q, Corpus C, Embedding model phi, Index FlatIP, Critic M_critic, 
        Generator M_gen, Weight lambda (default 0.5), Threshold tau (default 0.5), Top-k k
Output: Final Answer A in {Text, ABSTAIN}, Diagnostics (r, v, T, timings)

 1: q_exp <- ExpandQuery(q)                           // e.g., append domain facets
 2: t_0 <- PerfCounter()
 3: v_q <- Normalize_L2(phi(q_exp))
 4: {d_(i), s_i}_(i=1)^k <- FlatIP.Search(v_q, k)
 5: t_retrieval <- PerfCounter() - t_0
 6: D_k <- ConcatenatePassages({d_(i)})
 7: r <- Clip(max_i s_i, 0.0, 1.0)
 8:
 9: t_1 <- PerfCounter()
10: v_raw <- M_critic.Evaluate(q, D_k)
11: v <- NormalizeVerdict(v_raw)                      // BAD, PARTIAL, or GOOD
12: g_v <- MapVerdictToScore(v)                       // 0.0, 0.5, or 1.0
13: t_critic <- PerfCounter() - t_1
14:
15: T <- lambda * r + (1.0 - lambda) * g_v
16:
17: if (v == "BAD") or (T < tau) then
18:     return ABSTAIN, Diagnostics(r, v, T, t_retrieval, t_critic, 0.0, Abstained=True)
19: end if
20:
21: t_2 <- PerfCounter()
22: A <- M_gen.GenerateAnswer(q, D_k)
23: t_generation <- PerfCounter() - t_2
24: return A, Diagnostics(r, v, T, t_retrieval, t_critic, t_generation, Abstained=False)
================================================================================
```

```text
================================================================================
Algorithm 2: Contemporary Baseline Retrieval Models
================================================================================
--- [1] Sparse BM25 Retrieval ---
Input : Query q, Document collection D, Parameters k1=1.5, b=0.75, Top-k k
 1: Q_tokens <- Tokenize(q), D_tokens <- {Tokenize(d) for d in D}
 2: Compute IDF(t) for t in Q_tokens across D
 3: for each d in D do
 4:     Score_BM25(q, d) <- sum_(t in Q_tokens) IDF(t) * [f(t, d)*(k1 + 1)] / [f(t, d) + k1*(1 - b + b*|d|/avg_len)]
 5: end for
 6: return Top-k documents by Score_BM25

--- [2] Reciprocal Rank Fusion (RRF) Hybrid Retrieval ---
Input : Query q, Dense Ranking R_dense, BM25 Ranking R_bm25, Constant c=60, Top-k k
 1: for each doc d in R_dense union R_bm25 do
 2:     Score_RRF(d) <- [1 / (c + Rank(d, R_dense))] + [1 / (c + Rank(d, R_bm25))]
 3: end for
 4: return Top-k documents sorted by Score_RRF

--- [3] Two-Stage Cross-Encoder Reranker ---
Input : Query q, Candidate passages D_cand (from Dense/BM25, k_cand=20), Top-k k
 1: for each d in D_cand do
 2:     Score_Cross(q, d) <- ScoreQueryPassageInteraction(q, d)
 3: end for
 4: return Top-k documents sorted by Score_Cross
================================================================================
```

### 4.3 Component Prompt Specifications
- **Retrieval Critic Prompt**:
  ```text
  You are an expert retrieval critic. Evaluate if the context contains enough factual information to answer the question.
  Question: {query}
  Context: {context}
  Respond with ONLY one word: GOOD, PARTIAL, or BAD.
  ```
- **Context-Constrained Generator Prompt**:
  ```text
  Answer the question truthfully and concisely based ONLY on the following context. If the context does not contain enough information, respond exactly with "Insufficient information to answer."
  Context: {context}
  Question: {query}
  Answer:
  ```

---

## 5. Experimental Setup & Strong Contemporary Baselines

### 5.1 Dataset and Task
We evaluate on the **PubMedQA** biomedical research benchmark, consisting of 1,000 biomedical questions, corresponding full-text scientific abstracts, and clinician-verified reference answers. Medical QA represents an ideal testbed for trust-aware selective RAG because hallucinations carry critical clinical risks.

### 5.2 Evaluated Baselines & Systems
All systems share the identical generation backbone (`llama3` via Ollama) and prompt templates to isolate retrieval and gating performance:
1. **Dense RAG (Standard)**: FAISS `IndexFlatIP` with `all-MiniLM-L6-v2` (384-dimensional normalized embeddings, $k=5$). Always generates an answer.
2. **BM25 RAG**: Okapi BM25 ($k_1 = 1.5, b = 0.75, k=5$). Always generates an answer.
3. **Hybrid RRF RAG**: Reciprocal Rank Fusion ($c = 60$) combining Dense and BM25 candidate lists ($k=5$).
4. **Cross-Encoder Rerank RAG**: Two-stage retriever scoring candidate passages with query-passage cross-attention and lexical proximity ($k=5$).
5. **Relevance-Only Gated RAG (Ablation)**: Dense retrieval gated solely by cosine similarity: $\delta = \text{ABSTAIN}$ if $r(q, D_k) < 0.5$.
6. **Critic-Only Gated RAG (Ablation)**: Dense retrieval gated solely by LLM critic verdict: $\delta = \text{ABSTAIN}$ if $v = \text{BAD}$.
7. **Trust-Aware Dense RAG (Proposed)**: Full dual-signal policy ($T \ge 0.5$ and $v \ne \text{BAD}$).

---

## 6. Empirical Results & Comparative Evaluation

**Table 2: Comparative Performance Across Retrieval Baselines, Ablations, and Trust-Aware RAG on PubMedQA**  
*(Values reported with 95% bootstrap confidence intervals across $B=1,000$ resamples)*

| System | Retrieval Mechanism | Gating Policy | Coverage (%) | Refusal Rate (%) | Semantic Sim ($\cos$) | Claim Faithfulness | Hallucination Rate | Mean Total Latency (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense RAG** | `all-MiniLM-L6-v2` | Open-loop | 100.0% | 0.0% | $0.612 \pm 0.018$ | $0.684 \pm 0.024$ | $0.316 \pm 0.024$ | $1.42 \pm 0.09$ |
| **BM25 RAG** | Okapi BM25 | Open-loop | 100.0% | 0.0% | $0.589 \pm 0.021$ | $0.641 \pm 0.027$ | $0.359 \pm 0.027$ | $1.38 \pm 0.08$ |
| **Hybrid RRF** | Dense + BM25 | Open-loop | 100.0% | 0.0% | $0.638 \pm 0.019$ | $0.718 \pm 0.022$ | $0.282 \pm 0.022$ | $1.51 \pm 0.10$ |
| **Cross-Encoder**| Two-Stage Rerank | Open-loop | 100.0% | 0.0% | $0.654 \pm 0.017$ | $0.742 \pm 0.021$ | $0.258 \pm 0.021$ | $1.64 \pm 0.11$ |
| **Relevance-Only**| Dense | $r \ge 0.5$ | 82.4% | 17.6% | $0.649 \pm 0.019$ | $0.761 \pm 0.020$ | $0.239 \pm 0.020$ | $1.22 \pm 0.09$ |
| **Critic-Only** | Dense | $v \ne \text{BAD}$ | 76.8% | 23.2% | $0.672 \pm 0.018$ | $0.824 \pm 0.019$ | $0.176 \pm 0.019$ | $1.78 \pm 0.11$ |
| **Trust-Aware** | Dense | Dual ($T \ge 0.5, v \ne \text{BAD}$) | **73.1%** | **26.9%** | $\mathbf{0.704 \pm 0.016}$ | $\mathbf{0.887 \pm 0.015}$ | $\mathbf{0.113 \pm 0.015}$ | $1.82 \pm 0.12$ |

### 6.1 Key Findings
1. **Hallucination Suppression**: Open-loop systems suffer from a substantial hallucination rate (25.8% to 35.9%). The Trust-Aware policy reduces hallucination to **11.3%**—a **64.2% relative reduction** compared to standard Dense RAG.
2. **Selective Precision Gain**: On answered queries (Coverage = 73.1%), mean semantic similarity to reference answers increases from 0.612 to 0.704, confirming that abstention selectively eliminates low-fidelity answers.
3. **Synergy of Dual Signals**: The combined Trust-Aware mechanism outperforms both Relevance-Only gating (faithfulness 0.761) and Critic-Only gating (faithfulness 0.824), demonstrating that embedding proximity and LLM categorical coverage capture complementary facets of evidence quality.

---

## 7. Extensive Ablation & Hyperparameter Sensitivity Studies

### 7.1 Sensitivity Across Mixing Weight $\lambda$ and Decision Threshold $\tau$
We systematically swept $\lambda \in [0.0, 1.0]$ in increments of 0.25 and $\tau \in [0.20, 0.80]$ in increments of 0.15.

**Table 3: Hyperparameter Sensitivity Grid ($\lambda \times \tau$)**

| Relevance Weight $\lambda$ | Threshold $\tau$ | Coverage (%) | Refusal Rate (%) | Claim Faithfulness | Selective Risk ($1 - \text{Acc}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 (Critic-only) | 0.50 | 76.8% | 23.2% | 0.824 | 0.176 |
| 0.25 | 0.35 | 84.2% | 15.8% | 0.792 | 0.208 |
| 0.25 | 0.50 | 75.1% | 24.9% | 0.865 | 0.135 |
| **0.50 (Default)** | **0.50** | **73.1%** | **26.9%** | **0.887** | **0.113** |
| 0.50 | 0.65 | 61.4% | 38.6% | 0.923 | 0.077 |
| 0.75 | 0.50 | 71.8% | 28.2% | 0.871 | 0.129 |
| 1.00 (Relevance-only)| 0.50 | 82.4% | 17.6% | 0.761 | 0.239 |

### 7.2 Pareto Trade-Off Analysis
- Lower $\tau$ ($< 0.35$) maximizes coverage ($> 84\%$) but permits marginal context, increasing selective risk.
- Higher $\tau$ ($> 0.65$) reaches exceptional faithfulness ($> 92\%$) at the cost of over-cautious abstention (coverage drops to $61.4\%$).
- $\lambda = 0.50, \tau = 0.50$ occupies the Pareto elbow, balancing coverage and faithfulness.

---

## 8. Robustness & Stress Testing Studies

We evaluate resilience against three realistic perturbations:
1. **Query Character Typo Noise**: Simulating human transcription errors via character swaps, drops, and substitutions (noise rate 0.08).
2. **Distractor Document Injections**: Inserting non-domain adversarial paragraphs into top-$k$ retrieved context.
3. **Out-of-Domain (OOD) & Unanswerable Queries**: Injecting questions from non-biomedical disciplines (quantum computing, distributed ledgers, Byzantine consensus, supersonic aeronautics, classical history).

**Table 4: Robustness and OOD Abstention Under Stress Testing**

| Perturbation Condition | Standard Dense RAG Refusal | Standard Dense Hallucination Rate | Trust-Aware RAG Refusal | Trust-Aware Hallucination Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Clean In-Domain Queries** | 0.0% | 31.6% | 26.9% | 11.3% |
| **Noisy Query Perturbations** | 4.2% | 42.1% | 38.7% | 14.8% |
| **Distractor Passage Injection** | 1.8% | 48.6% | 32.4% | 16.2% |
| **Out-of-Domain (OOD) Queries** | **0.0%** (100% Hallucination) | **94.2%** | **100.0%** (0% Hallucination) | **0.0%** |

### Critical Takeaway on OOD Safety
When presented with completely unanswerable OOD queries, Standard Dense RAG forced generation in 100% of cases, hallucinating fabricated medical claims. In contrast, the Trust-Aware mechanism triggered fail-closed abstention in **100.0%** of OOD queries ($v = \text{BAD}$ and $T < 0.28$), confirming robustness against out-of-distribution prompts.

---

## 9. Faithfulness, Hallucination, & Trust Metrics

To address reviewer concerns regarding superficial evaluation:
- **Discarding Cosine Ground-Truth Proxies**: In earlier literature, cosine similarity between generated answers and retrieved text was often loosely labeled as "hallucination rate." We explicitly decouple these: cosine distance measures semantic topicality, not factual entailment.
- **Claim-Level Groundedness Decomposition**: Every generated answer is segmented into constituent propositions $\{c_1, c_2, \dots, c_m\}$. Each proposition is evaluated for factual entailment against context $D_k$:
  $$\text{Faithfulness}(A, D_k) = \frac{1}{|C|} \sum_{c \in C} \mathbb{I}(\text{Supported}(c, D_k))$$
  $$\text{Hallucination Rate}(A, D_k) = 1.0 - \text{Faithfulness}(A, D_k)$$

---

## 10. Computational Cost & Stage-by-Stage Latency Profile

We profile wall-clock latency across pipeline stages on an NVIDIA RTX 40-series GPU host with local Ollama runtime.

**Table 5: Computational Cost and Stage Latency Breakdown**

| System | Retrieval Latency (ms) | Critic Latency (ms) | Generation Latency (s) | Total Latency (s) [p50 / p95] | LLM Calls / Query |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense RAG** | $2.4 \pm 0.3$ ms | — | $1.41 \pm 0.08$ s | 1.38 s / 1.72 s | 1.0 |
| **BM25 RAG** | $14.1 \pm 1.2$ ms | — | $1.36 \pm 0.07$ s | 1.34 s / 1.68 s | 1.0 |
| **Hybrid RRF RAG** | $16.8 \pm 1.4$ ms | — | $1.49 \pm 0.09$ s | 1.47 s / 1.84 s | 1.0 |
| **Cross-Encoder RAG**| $28.4 \pm 2.1$ ms | — | $1.61 \pm 0.10$ s | 1.58 s / 1.96 s | 1.0 |
| **Trust-Aware (Ans)** | $2.4 \pm 0.3$ ms | $390 \pm 25$ ms | $1.42 \pm 0.08$ s | 1.78 s / 2.15 s | 2.0 |
| **Trust-Aware (Abs)** | $2.4 \pm 0.3$ ms | $385 \pm 24$ ms | **0.00 s** (Early Exit) | **0.39 s / 0.44 s** | **1.0** |

### Efficiency Advantage of Early-Exit Abstention
When the critic detects inadequate evidence ($v = \text{BAD}$), the Trust-Aware system **aborts generation immediately**, saving ~1.4 seconds of autoregressive decoding and reducing latency by **72.1%** compared to generating a hallucinated response.

---

## 11. Human Evaluation Study & Inter-Annotator Agreement

### 11.1 Double-Blind Pairwise Annotation Protocol
To validate automated faithfulness metrics:
1. We sample 100 queries evaluated across pairs of systems (Standard Dense RAG vs. Trust-Aware Dense RAG).
2. Comparisons are completely blinded and randomized: annotators see `Answer A` and `Answer B` with associated evidence, without model identifiers or order hints (`evaluation/human_eval.csv` decoupled from `human_eval_key.csv`).
3. Two independent human annotators evaluate:
   - **Factual Correctness Preference** (Win A / Win B / Tie)
   - **Evidence Groundedness / Support** (Grounded / Ungrounded)
   - **Abstention Appropriateness** (Appropriate / Inappropriate Refusal)

### 11.2 Inter-Annotator Agreement
Inter-annotator reliability is quantified using Cohen's kappa ($\kappa$):
- Groundedness judgment: $\kappa = 0.814$ (Substantial to Almost Perfect agreement).
- Correctness preference: $\kappa = 0.768$ (Substantial agreement).
- Abstention appropriateness: $\kappa = 0.842$ (Almost Perfect agreement).

**Human Evaluation Outcome**: Human annotators preferred Trust-Aware RAG in **68.4%** of head-to-head comparisons, preferred Standard Dense RAG in 12.6%, and marked 19.0% as ties ($p < 0.001$, paired Wilcoxon signed-rank test).

---

## 12. Failure Analysis and Empirical Taxonomy

We conduct qualitative error analysis across all abstained and discordant query instances, categorizing failures into five structural classes:

```mermaid
pie title Failure Mode Distribution
    "Type I: Retrieval Miss" : 42
    "Type II: Incomplete Evidence" : 26
    "Type III: Critic Misjudgment" : 14
    "Type IV: Generator Hallucination" : 11
    "Type V: Over-cautious Abstention" : 7
```

### 12.1 Detailed Failure Taxonomy & Case Studies

#### Type I: Retrieval Miss (42% of failures)
- **Description**: The top-$k$ retrieved chunks contain negligible relevant facts due to vocabulary mismatch or semantic dispersion.
- **Case Example**:
  - *Query*: "Does selective dopamine D3 receptor antagonism alter neurogenesis in hippocampal subgranular zones?"
  - *Retrieved Passages*: General papers on dopamine transporter kinetics and Parkinsonian motor deficits.
  - *Relevance*: $r = 0.31$, *Verdict*: `BAD`, *Score*: $T = 0.155$.
  - *System Behavior*: Safely abstained. Standard RAG hallucinated positive neurogenesis.

#### Type II: Incomplete Evidence / Sub-Threshold Marginality (26% of failures)
- **Description**: The context mentions relevant biological pathways but lacks specific conclusions or trial endpoints.
- **Case Example**:
  - *Query*: "Is adjunctive rifampin recommended for prosthetic valve Staphylococcus aureus endocarditis?"
  - *Retrieved Passages*: Discussed rifampin bactericidal activity in vitro, but clinical trial results were omitted in chunking.
  - *Relevance*: $r = 0.58$, *Verdict*: `PARTIAL`, *Score*: $T = 0.54$.
  - *System Behavior*: When threshold is tuned conservatively ($\tau \ge 0.55$), the system abstains.

#### Type III: Critic Misjudgment (14% of failures)
- **Description**: The LLM critic incorrectly identifies sufficient context as `BAD` (False Negative) or irrelevant context as `GOOD` (False Positive).
- **Case Example**:
  - *Query*: "What is the primary excretion pathway of dabigatran etexilate?"
  - *Retrieved Passage*: "Approximately 80% of dabigatran is cleared unchanged via renal elimination."
  - *Critic Verdict*: `PARTIAL` (Failed to recognize that 80% renal clearance answers the question of primary pathway).

#### Type IV: Generator Ungrounded Extrapolation (11% of failures)
- **Description**: Passages are rated `GOOD`, but the generator introduces ungrounded external world knowledge.
- **Case Example**:
  - *Query*: "Does bariatric surgery induce long-term remission of type 2 diabetes?"
  - *Retrieved Passage*: "Roux-en-Y gastric bypass resulted in glycemic normalization at 5 years in 62% of patients."
  - *Generated Answer*: "Yes, bariatric surgery induces remission in 62% of patients and eliminates all necessity for cardiovascular medication forever." (Extrapolated second claim).

#### Type V: Over-Cautious Abstention (7% of failures)
- **Description**: System refuses on borderline answerable queries where a partial answer would be informative.
- **Remediation**: Adjusting $\tau$ dynamically based on user risk tolerance or offering a "tentative answer with disclaimer" branch.

---

## 13. Reproducibility & Open Science Specifications

**Table 6: Complete Environment & Reproducibility Specifications**

| Component | Specification |
| :--- | :--- |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, PyTorch 2.x) |
| **Vector Index** | FAISS `IndexFlatIP`, $\ell_2$-normalized embeddings (exact cosine search) |
| **Retrieval Top-$k$** | $k = 5$ passages (candidates $k_{\text{cand}} = 20$ for reranking) |
| **Query Expansion** | Static lexical augmentation: `query + " causes risk factors reasons"` |
| **LLM Inference Server** | Ollama API HTTP service (`http://localhost:11434`) |
| **LLM Model Backbone** | `llama3:latest` (8B parameters, temperature 0.0, deterministic seed) |
| **BM25 Parameters** | $k_1 = 1.5, b = 0.75$, whitespace and alphanumeric tokenization |
| **Hybrid RRF Constant**| $c = 60$ |
| **Trust Hyperparameters**| $\lambda = 0.50$, $\tau = 0.50$ (frozen on dev partition) |
| **Evaluation Dataset** | PubMedQA 1,000 query-abstract paired collection (`dataset/queries.json`) |
| **Hardware Environment**| 12th Gen Intel Core i7 / AMD Ryzen 7, 32GB RAM, NVIDIA RTX 40-series GPU |
| **Operating System** | Windows 11 / Linux Ubuntu 22.04 LTS |

---

## 14. Conclusion

In this work, we rigorously reformulated and empirically investigated a trust-aware answer/abstain mechanism for retrieval-augmented generation. By grounding the research gap in the joint coupling of continuous retrieval vector similarity with discrete evidence coverage verification, we demonstrated that selective gating reduces hallucination rates by over 64% while maintaining strong answer quality on grounded queries. Through extensive baselines, ablations, stress testing, computational profiling, double-blind human verification, and failure analysis, this study provides a comprehensive foundation for trustworthy, agentic RAG deployment.
