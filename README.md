# Controlled Agentic RAG System

A research prototype for document-grounded question answering with a trust-aware answer/abstain policy and an extensible evaluation harness.

## ✨ Key Features

### Core Functionality
- 📤 **Document Upload**: Support for PDF, TXT, DOCX formats
- 🔍 **Intelligent Retrieval**: FAISS-based semantic search with query enhancement
- 🤖 **Agentic Reasoning**: Retrieval critic and context-grounded answer generation
- ✅ **Trust Gate**: Combines dense relevance and critic verdict before generation

### Evaluation & Monitoring  
- 📊 **Real-time Performance**: Execution time tracking (retrieval, critic, generation)
- 📈 **Contemporary RAG Baselines**: Dense, BM25, Hybrid RRF, Cross-Encoder reranking, and gated ablations
- 🎯 **Evaluation Metrics**: Claim-level faithfulness, hallucination rates, selective QA coverage/risk, and reference similarity
- 🧪 **Robustness & Stress Testing**: Perturbation noise, distractor injection, and out-of-domain (OOD) tests
- 📋 **Reproducible Artifacts**: Full manuscript revision (`REVISED_MANUSCRIPT.md`), point-by-point reviewer rebuttal (`RESPONSE_TO_REVIEWERS.md`), query-level CSVs, and blinded human-evaluation templates

### User Experience
- 🎨 **Modern Web UI**: Clean, responsive Flask interface
- ⏱️ **Timing Display**: Real-time execution time shown on frontend
- 🗑️ **Document Management**: Easy upload/delete interface
- 📌 **No Clutter**: Only shows final answer (no intermediate chunk info)

## 📊 System Architecture

```
User Input
    ↓
[Upload Documents] → Chunking → FAISS Index
    ↓
[Query] → Retrieval (with timing)
    ↓
[Retrieval Critic] → Validates context
    ↓
[Agent Reasoning] (with timing) → Generate answer
    ↓
[Final Answer + Execution Time]
    ↓
Display on Web UI
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Web Interface
```bash
python app.py
```
Visit: `http://localhost:5000`

### 3. Evaluate System
```bash
# Compare dense, BM25, hybrid, and trust-aware RAG
python evaluation/run_revision_eval.py --limit 30 --seed 13

# Generate plots from measured results
python evaluation/plot.py
```

## 📊 Evaluation Metrics

| Metric | Description | Caveat |
|--------|-------------|--------|
| **Reference similarity** | Embedding similarity to the supplied answer | Not a correctness judgment |
| **Answer-context similarity** | Embedding similarity proxy | Not a hallucination/faithfulness measure |
| **Latency** | Retrieval, critic, generation, and total wall time | Report hardware and model versions |
| **Refusal rate** | Fraction of queries with abstention | Interpret with answer quality/coverage |

## 📈 Generated Plots

`evaluation/plot.py` generates descriptive quality, refusal, latency, and trust-index figures from measured rows in `evaluation/revision_results.csv`. These plots do not estimate hallucination, OOD robustness, calibration, or distributed scaling.

## 🎯 Use Cases

- **QA Systems**: Document-based question answering
- **Customer Support**: Automated agent-based support
- **Research**: Controlled information retrieval and synthesis
- **Education**: Interactive learning from documents

## 📁 Project Structure

```
├── app.py                    # Flask web application
├── core/
│   ├── agent.py              # Agentic RAG logic with timing
│   ├── state.py              # State management + timing tracking
│   ├── budget.py
│   └── controller.py
├── retrieval/
│   ├── retriever.py          # Semantic search
│   ├── critic.py             # Quality validation
│   └── store.py              # FAISS index management
├── reasoning/
│   └── generator.py          # Answer generation
├── evaluation/
│   ├── run_revision_eval.py  # Dense/BM25/hybrid/cross-encoder/trust-aware comparison
│   ├── robustness_eval.py    # Typo noise, distractor, and OOD evaluations
│   ├── ablation_study.py     # Hyperparameter grid sensitivity sweep
│   ├── failure_analysis.py   # Five-class automated failure categorization
│   ├── create_human_eval.py  # Blinded pairwise annotation sheets
│   ├── plot.py               # Measured quality, latency, and faithfulness plots
│   └── metrics.py            # Faithfulness, hallucination, and selective QA metrics
├── tests/
│   └── test_trust_policy.py  # 13 automated unit tests
├── templates/
│   └── index.html            # Clean, modern web UI
├── dataset/                  # Documents and queries
├── REVISED_MANUSCRIPT.md     # Full publication manuscript draft
├── RESPONSE_TO_REVIEWERS.md  # Detailed point-by-point reviewer rebuttal
├── EVALUATION_GUIDE.md       # Comprehensive evaluation reproduction guide
├── MANUSCRIPT_REVISION.md    # Revision overview and protocol
└── QUICK_START.md            # Quick reference guide
```

## 🔧 Configuration

### LLM Settings
Edit `utils/llm.py` to configure:
- Model selection
- Temperature
- Max tokens
- API keys

### Retrieval Parameters
Edit `retrieval/retriever.py`:
- `top_k`: Number of documents to retrieve
- Enhanced query construction
- Similarity threshold

### Evaluation Settings
Edit `evaluation/run_revision_eval.py`:
- Number of queries to evaluate
- OOD test queries
- Similarity threshold for hallucination

## ⏱️ Timing Features

The system now tracks:
- **Total Execution Time**: Complete pipeline duration
- **Retrieval Time**: Document search latency
- **Generation Time**: Answer generation latency
- **Per-Query Timing**: Individual query performance

Timing is:
- Displayed on web UI after each query
- Logged during evaluation
- Included in all metrics reports
- Visualized in execution time plots

## 📊 Evaluation Commands

```bash
# Run a reproducible 30-query evaluation sample
python evaluation/run_revision_eval.py --limit 30 --seed 13

# Run the full 1,000-query dataset
python evaluation/run_revision_eval.py --limit 1000 --seed 13

# Generate visualization plots
python evaluation/plot.py

# View results in a spreadsheet
evaluation/revision_results.csv

# Human comparison template
python evaluation/create_human_eval.py --seed 13
```

## 🎨 Frontend Improvements

Recent updates include:
- ✨ Modern gradient background
- 📝 Clear visual hierarchy
- ⏱️ Real-time execution timer
- 🎯 Focus on output only (no intermediate steps)
- 📱 Responsive design
- 🎨 Color-coded buttons and elements

## 🧪 Testing

### Manual Testing
1. Upload a document
2. Ask a question
3. Check if:
   - Answer displays correctly
   - Execution time shows
   - No errors in console

### Automated Evaluation
```bash
python evaluation/run_revision_eval.py --limit 30 --seed 13
```

The evaluation harness records retrieval/generation outputs and latency for four RAG configurations. It does not complete the required human, faithfulness, OOD, robustness, or statistical studies; see `MANUSCRIPT_REVISION.md`.

## 📝 Files Modified in Latest Update

- ✅ `templates/index.html` - Enhanced UI, execution time display
- ✅ `core/state.py` - Added timing attributes
- ✅ `core/agent.py` - Added timing measurements
- ✅ `app.py` - Capture and display execution time
- `core/trust_policy.py` - Formal trust decision score and abstention gate
- `evaluation/run_revision_eval.py` - Reproducible baseline comparison
- `evaluation/create_human_eval.py` - Blinded pairwise annotation-sheet generator
- `MANUSCRIPT_REVISION.md` - Novelty positioning, method, and outstanding studies

## 🚀 Next Steps

1. **Run the evaluation** to benchmark your system
2. **Review the plots** to identify improvements
3. **Adjust parameters** based on results
4. **Deploy** to production with monitoring

## 📞 Support

For detailed information on:
- **Metrics**: See `EVALUATION_GUIDE.md`
- **Quick Setup**: See `QUICK_START.md`
- **Code Logic**: Check inline comments in each module

## 📄 License

[Add your license here]

## 🙏 Contributing

Contributions welcome! Please:
1. Test changes locally
2. Update documentation
3. Ensure evaluation passes
4. Submit with metrics

---

**Last Updated**: May 2026
**Status**: Research prototype; manuscript-level claims require the studies in `MANUSCRIPT_REVISION.md`.
**Maintenance**: Active