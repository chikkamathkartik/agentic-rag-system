from baseline import baseline_rag
from core.agent import run_agent
from core.state import State


# ---------------- FORMATTER ----------------
def format_output(memory):

    sections = {
        "definition": [],
        "cause": [],
        "condition": [],
        "prevention": [],
        "example": [],
        "other": []
    }

    def classify(text):
        t = text.lower()

        if " is " in t:
            return "definition"
        if "cause" in t or "occur" in t:
            return "cause"
        if "mutual exclusion" in t or "circular wait" in t:
            return "condition"
        if "prevent" in t or "avoid" in t:
            return "prevention"
        if "example" in t:
            return "example"

        return "other"

    for m in memory:
        topic = classify(m["text"])
        sections[topic].append(m["text"])

    return sections


# ---------------- COMPARISON ----------------
def run_comparison(query):

    print("\n==============================")
    print(f"Query: {query}")
    print("==============================")

    # -------- BASELINE --------
    print("\n--- BASELINE OUTPUT ---")
    baseline_output = baseline_rag(query)
    print(baseline_output)

    # -------- AGENT --------
    print("\n--- AGENT OUTPUT ---")
    state = State(query=query)
    memory = run_agent(state)

    # -------- FORMATTED OUTPUT --------
    sections = format_output(memory)

    print("\n====== FINAL ANSWER ======\n")

    if sections["definition"]:
        print("Definition:")
        for s in sections["definition"]:
            print("-", s)
        print()

    if sections["cause"] or sections["condition"]:
        print("Causes / Conditions:")
        for s in sections["cause"] + sections["condition"]:
            print("-", s)
        print()

    if sections["prevention"]:
        print("Prevention:")
        for s in sections["prevention"]:
            print("-", s)
        print()

    if sections["example"]:
        print("Example:")
        for s in sections["example"]:
            print("-", s)
        print()

    if sections["other"]:
        print("Additional Info:")
        for s in sections["other"]:
            print("-", s)
        print()


# ---------------- MAIN ----------------
if __name__ == "__main__":

    queries = [
        "Explain deadlock",
        "What is livelock",
        "What causes deadlock",
        "Why does livelock occur",
        "How to prevent deadlock",
        "How to avoid livelock",
        "Explain deadlock with causes and prevention",
        "Difference between deadlock and livelock with example"
    ]

    for q in queries:
        run_comparison(q)