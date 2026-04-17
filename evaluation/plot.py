import matplotlib.pyplot as plt

def plot_history(history):

    steps = [h["step"] for h in history]
    memory = [h["memory_size"] for h in history]
    confidence = [h["confidence"] for h in history]

    # 🔹 Plot 1: Memory Growth
    plt.figure()
    plt.plot(steps, memory, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Memory Size")
    plt.title("Knowledge Growth Over Steps")
    plt.grid()
    plt.show()

    # 🔹 Plot 2: Confidence Growth
    plt.figure()
    plt.plot(steps, confidence, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Confidence")
    plt.title("Confidence Growth Over Steps")
    plt.grid()
    plt.show()