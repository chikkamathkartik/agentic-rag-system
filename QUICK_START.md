# 🚀 Quick Start Guide - Agentic RAG System

## One-Minute Setup

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Run the Web Interface**
```bash
python app.py
```
Visit: `http://localhost:5000`

---

## 📊 Quick Evaluation (5 minutes)

### Run Everything in Sequence
```bash
# 1. Run evaluation (compares baseline vs agent RAG)
python evaluation/run_evaluation.py

# 2. Generate all plots
python evaluation/plot.py

# 3. View results
#   - Check: evaluation/results.csv
#   - View: graphs/ (8 PNG files)
```

---

## ✨ What's New in This Update

✅ **Frontend Improvements:**
- Clean, modern interface
- Only shows final answer (no ugly chunk info)
- **NEW**: Displays execution time for each query

✅ **Timing Features:**
- Total execution time: Input → Output
- Retrieval time breakdown
- Generation time breakdown
- Time statistics in evaluation reports

✅ **Enhanced Evaluation:**
- 8 comprehensive plots instead of basic charts
- Timing metrics for both baseline and agent
- Per-query improvement analysis
- Performance heatmap summary

✅ **Better Visualizations:**
- High-quality 300 DPI plots
- Color-coded comparisons
- Statistical confidence bands
- Trade-off analysis charts

---

## 📊 Key Files to Know

| File | Purpose |
|------|---------|
| `app.py` | Web interface - run to start frontend |
| `core/agent.py` | Main agentic RAG logic with timing |
| `core/state.py` | State management + timing tracking |
| `evaluation/run_evaluation.py` | Benchmark evaluation script |
| `evaluation/plot.py` | Generate 8 comprehensive plots |
| `templates/index.html` | Clean, modern web UI |
| `EVALUATION_GUIDE.md` | Detailed metrics explanation |

---

## 🎯 Metrics at a Glance

```
Similarity Score (0-1)          ← How well answer matches truth
Hallucination Rate (0-1)        ← Fabricated information %
Execution Time (seconds)        ← How fast is the system
Response Length (words)         ← Conciseness of answers
Refusal Rate (0-1)              ← When agent admits uncertainty
OOD Hallucination (0-1)         ← Mistakes on unknown topics
```

---

## 🔧 Common Commands

```bash
# Run the web app
python app.py

# Evaluate the system
python evaluation/run_evaluation.py

# Generate plots
python evaluation/plot.py

# Run a single query via CLI
python main.py

# View evaluation results
cat evaluation/results.csv
```

---

## 📈 Check Your Plots

After running `evaluation/plot.py`, check these files:

1. `1_similarity_trend.png` - See Agent vs Baseline performance
2. `2_hallucination_comparison.png` - How safe is the system?
3. `3_execution_time.png` - Speed breakdown
4. `4_metrics_comparison.png` - Overall performance
5. `7_query_by_query_comparison.png` - Which queries improved most?
6. `8_performance_heatmap.png` - Summary view

---

## ✅ Success Checklist

- [ ] Web app runs without errors
- [ ] Can upload documents and ask questions
- [ ] Execution time displays on web UI
- [ ] Evaluation script completes without errors
- [ ] Plots are generated in `graphs/`
- [ ] Agent outperforms baseline on similarity
- [ ] Hallucination rate is lower for agent

---

## 🆘 Troubleshooting

**Web app won't start:**
```bash
# Check if port 5000 is in use
# Or run on different port:
python app.py --port 5001
```

**No evaluation results:**
```bash
# Ensure queries.json exists in dataset/
# Ensure documents are in dataset/docs/
# Check baseline.py is available
```

**Plots not generating:**
```bash
# Ensure results.csv exists
python evaluation/run_evaluation.py  # Run this first
python evaluation/plot.py            # Then this
```

---

## 📞 Need More Info?

- See `EVALUATION_GUIDE.md` for detailed metric explanations
- Check code comments in individual files
- Review generated plots for visual insights

Happy evaluating! 🎉
