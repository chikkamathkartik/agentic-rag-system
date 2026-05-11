# 🤖 Agentic RAG System - Evaluation Guide

## Overview

This document explains how to evaluate the Agentic RAG system, understand the metrics, and interpret the comprehensive visualizations generated.

---

## 📊 Running the Evaluation

### Step 1: Prepare the Dataset
Ensure you have the following:
- `dataset/queries.json` - Contains questions and ground truth answers
- `dataset/docs/` - Document files to be indexed

### Step 2: Run Evaluation Script
```bash
python evaluation/run_evaluation.py
```

This will:
- ✅ Run 30 queries through both baseline RAG and agentic RAG
- ✅ Evaluate on OOD (Out-of-Distribution) queries
- ✅ Calculate metrics for each query
- ✅ Save results to `evaluation/results.csv`
- ✅ Print detailed metrics to console

### Step 3: Generate Plots
```bash
python evaluation/plot.py
```

This generates 8 comprehensive visualizations in `graphs/`

---

## 📈 Key Metrics Explained

### 1. **Similarity Score** ⭐
- **What**: Cosine similarity between model output and ground truth answer
- **Range**: 0 to 1 (higher is better)
- **Interpretation**:
  - > 0.7 = Very good match
  - 0.5-0.7 = Good match
  - 0.3-0.5 = Moderate match
  - < 0.3 = Poor match

### 2. **Hallucination Rate** 🚨
- **What**: Percentage of responses that contain fabricated information
- **Range**: 0 to 1 (lower is better)
- **Calculation**: Detected when similarity between answer and context chunks < 0.15
- **Target**: < 10% is acceptable, < 5% is excellent

### 3. **Execution Time** ⏱️
- **Total Time**: Complete time from query to final answer
- **Retrieval Time**: Time spent searching the document database
- **Generation Time**: Time spent generating the response
- **Target**: < 2s per query for real-time applications

### 4. **Response Length** 📝
- **What**: Average number of words in responses
- **Interpretation**: Longer isn't always better; balanced length is ideal
- **Comparison**: Agent vs Baseline response conciseness

### 5. **Refusal Rate** 🙅
- **What**: Percentage of times the agent refuses to answer
- **When**: When insufficient information is found in context
- **Good**: Indicates the system won't hallucinate when it shouldn't

### 6. **OOD (Out-of-Distribution) Hallucination** 🌍
- **What**: Hallucination rate on queries outside the document scope
- **Importance**: Shows the system's ability to admit uncertainty
- **Target**: Agent should have LOWER OOD hallucination than baseline

---

## 📊 Generated Plots

### 1️⃣ **Similarity Trend** (`1_similarity_trend.png`)
- Shows similarity scores across all queries
- Shaded regions show which system performs better
- **What to look for**: Agent (blue) should trend above Baseline (red)

### 2️⃣ **Hallucination Comparison** (`2_hallucination_comparison.png`)
- Bar chart comparing hallucination rates
- **What to look for**: Agent bar should be significantly lower

### 3️⃣ **Execution Time** (`3_execution_time.png`)
- **Left plot**: Time breakdown (retrieval vs generation) per query
- **Right plot**: Pie chart showing average time distribution
- **What to look for**: Retrieval shouldn't dominate generation time

### 4️⃣ **Metrics Comparison** (`4_metrics_comparison.png`)
- Side-by-side comparison of key metrics
- **What to look for**: Agent bars should be better (higher similarity, lower hallucination)

### 5️⃣ **Response Length Distribution** (`5_response_length.png`)
- Histogram showing response length patterns
- **What to look for**: Similar distributions or appropriate length differences

### 6️⃣ **Similarity vs Hallucination Trade-off** (`6_similarity_hallucination_tradeoff.png`)
- Scatter plot showing the relationship
- **Ideal**: Cluster in top-left (high similarity, low hallucination)

### 7️⃣ **Query-by-Query Comparison** (`7_query_by_query_comparison.png`)
- Bar charts showing per-query improvements
- **What to look for**: Positive bars = Agent is better

### 8️⃣ **Performance Heatmap** (`8_performance_heatmap.png`)
- Summary heatmap of all key metrics
- **What to look for**: Greener colors indicate better performance

---

## 📋 Console Output Metrics

When running evaluation, you'll see:

```
===== FINAL METRICS =====

📊 Similarity Scores:
  Baseline: 0.5234 (±0.1823)    <- Average and std deviation
  Agent: 0.6891 (±0.1456)
  Improvement: 0.1657            <- How much agent improved

🚨 Hallucination Rates:
  Baseline: 0.2456
  Agent: 0.0987
  Reduction: 0.1469              <- How much hallucination decreased

⏱️ Execution Time:
  Baseline avg: 1.23s
  Agent avg total: 1.87s
    - Retrieval: 0.45s
    - Generation: 0.78s

📝 Response Length:
  Baseline avg: 125 words
  Agent avg: 98 words

🙅 Refusal Rate (Agent): 0.0667  <- How often agent refuses
🌍 OOD Hallucination - Baseline: 0.3456, Agent: 0.0987
```

---

## 🎯 Success Criteria

For a well-functioning Agentic RAG system:

| Metric | Target | Acceptable |
|--------|--------|-----------|
| Similarity Improvement | > 15% | > 5% |
| Hallucination Reduction | > 50% | > 20% |
| Execution Time | < 2s | < 5s |
| OOD Hallucination | < 10% | < 25% |
| Refusal Rate | 5-15% | 0-30% |

---

## 🔍 Troubleshooting

### **High Hallucination Rate**
- ✅ Check if documents are properly indexed
- ✅ Verify retrieval is finding relevant chunks
- ✅ Review the critic's verdicts

### **Slow Execution**
- ✅ Check retrieval latency (embedding search)
- ✅ Verify LLM API response time
- ✅ Consider using smaller embeddings model

### **Low Similarity Scores**
- ✅ Ensure documents are relevant to queries
- ✅ Check if ground truth answers are accurate
- ✅ Review the LLM prompt

### **High Refusal Rate**
- ✅ May be working as intended (better safe than sorry)
- ✅ Check if retrieval critic is too strict
- ✅ Increase threshold for "BAD" retrieval verdict

---

## 🚀 Frontend Integration

The web interface (Flask app) now displays:
- ✅ Query input form
- ✅ Generated answer
- ✅ ⏱️ **Execution time in seconds** (NEW!)
- ✅ Clean, modern UI (no more chunk clutter)

### Running the Web App
```bash
python app.py
```

Navigate to `http://localhost:5000`

---

## 📊 Results Files

- `evaluation/results.csv` - Raw metrics for all queries
- `graphs/` - All generated visualizations
  - 8 PNG files at 300 DPI for publication quality

---

## 💡 Tips for Best Results

1. **Diverse Queries**: Use questions covering different topics and complexity levels
2. **Accurate Ground Truth**: Ensure ground truth answers are factually correct
3. **Sufficient Documents**: More relevant documents = better retrieval
4. **Monitor OOD**: Regularly test on out-of-distribution queries
5. **Iterate**: Use plots to identify bottlenecks and improve iteratively

---

## 📞 Questions?

Check the code comments in:
- `core/agent.py` - Agent logic
- `evaluation/run_evaluation.py` - Evaluation metrics
- `evaluation/plot.py` - Visualization logic
