# Controlled Agentic RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with built-in quality control, comprehensive evaluation, and detailed performance tracking.

## ✨ Key Features

### Core Functionality
- 📤 **Document Upload**: Support for PDF, TXT, DOCX formats
- 🔍 **Intelligent Retrieval**: FAISS-based semantic search with query enhancement
- 🤖 **Agentic Reasoning**: Multi-step reasoning with built-in verification
- ✅ **Quality Control**: Retrieval critic validates context before generation
- 🧠 **Memory Management**: Confidence-based knowledge storage

### Evaluation & Monitoring  
- 📊 **Real-time Performance**: Execution time tracking (retrieval + generation)
- 📈 **8 Comprehensive Plots**: Detailed visualization of metrics
- 🎯 **Multiple Metrics**: Similarity, hallucination detection, response quality
- 🌍 **OOD Testing**: Out-of-distribution query evaluation
- 📋 **Detailed Reports**: CSV exports of all metrics

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
# Run comprehensive evaluation
python evaluation/run_evaluation.py

# Generate plots
python evaluation/plot.py
```

## 📊 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Similarity Score** | How well answer matches ground truth (0-1) | > 0.65 |
| **Hallucination Rate** | % of fabricated information (0-1) | < 10% |
| **Execution Time** | Total time from query to answer | < 2s |
| **Response Length** | Word count of responses | Varied |
| **Refusal Rate** | % of uncertain answers | 5-15% |
| **OOD Hallucination** | Hallucination on unknown topics | < 10% |

## 📈 Generated Plots

After running evaluation, you get:
1. **Similarity Trend** - Performance across queries
2. **Hallucination Comparison** - Safety metrics
3. **Execution Time** - Speed breakdown (retrieval vs generation)
4. **Metrics Comparison** - Side-by-side performance
5. **Response Length** - Distribution analysis
6. **Trade-off Analysis** - Similarity vs Hallucination
7. **Per-Query Improvements** - Query-by-query gains
8. **Performance Heatmap** - Summary overview

All plots saved as high-quality 300 DPI PNG files.

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
│   ├── run_evaluation.py     # Benchmark script with timing metrics
│   ├── plot.py               # 8 comprehensive visualizations
│   └── metrics.py
├── templates/
│   └── index.html            # Clean, modern web UI
├── dataset/                  # Documents and queries
├── graphs/                   # Generated visualizations
├── EVALUATION_GUIDE.md       # Detailed metrics documentation
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
Edit `evaluation/run_evaluation.py`:
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
# Run full evaluation (30 queries)
python evaluation/run_evaluation.py

# Generate visualization plots
python evaluation/plot.py

# View results
head evaluation/results.csv

# Check execution times
grep "time" evaluation/results.csv
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
python evaluation/run_evaluation.py
```

Checks:
- Baseline vs Agent performance
- Hallucination detection
- Response quality
- Execution efficiency

## 📝 Files Modified in Latest Update

- ✅ `templates/index.html` - Enhanced UI, execution time display
- ✅ `core/state.py` - Added timing attributes
- ✅ `core/agent.py` - Added timing measurements
- ✅ `app.py` - Capture and display execution time
- ✅ `evaluation/run_evaluation.py` - Enhanced metrics with timing
- ✅ `evaluation/plot.py` - 8 new comprehensive plots
- ✅ `EVALUATION_GUIDE.md` - New detailed documentation
- ✅ `QUICK_START.md` - New quick reference guide

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
**Status**: Production Ready
**Maintenance**: Active