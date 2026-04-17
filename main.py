# main.py

from core.state import State
from core.agent import run_agent
from retrieval.retriever import initialize_retriever

query = input("Enter your query: ")
state = State(query)
initialize_retriever("data/deadlock.txt")

result = run_agent(state)

print("\nFinal Output:\n")
for r in result:
    print("-", r["text"])
    
from evaluation.plot import plot_history

plot_history(state.history)