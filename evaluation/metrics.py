def hallucination_rate(memory):
    # naive metric
    return sum(1 for m in memory if len(m["text"]) < 20) / len(memory)

def knowledge_growth(memory_sizes):
    return memory_sizes[-1] - memory_sizes[0]

def efficiency_score(steps, tokens):
    return steps / (tokens + 1)