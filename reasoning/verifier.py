from utils.llm import call_llm

def verify(state, reasoning):

    prompt = f"""
    Question: {state.query}
    Context: {state.context}
    Answer: {reasoning}

    Rate correctness from 0 to 1.
    Only output a number like 0.7
    """

def verify(state, reasoning):

    if not reasoning:
        return 0.0

    if "error" in reasoning.lower():
        return 0.0

    # simple quality checks
    if len(reasoning.split(".")) < 2:
        return 0.3

    if len(reasoning) < 50:
        return 0.4

    #  accept reasonable outputs
    return 0.7