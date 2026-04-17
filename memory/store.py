import re
import difflib


# ---------------- CLEAN TEXT ----------------
def clean_text(text):

    remove_phrases = [
        "Based on the following context:",
        "Explain the concept clearly",
        "Give only the answer",
        "Answer the question clearly and concisely",
        "Question:",
        "Context:",
        "No extra instructions",
        "Here's the answer:",
        "Here's the output:"
    ]

    for phrase in remove_phrases:
        text = text.replace(phrase, "")

    text = text.replace("\n", " ")
    text = re.sub(r"missing topics[:\-]*", "", text.lower())
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------- EXTRACT KNOWLEDGE ----------------
def extract_knowledge(text):

    lines = re.split(r'\.\s+', text)
    facts = []

    for line in lines:
        line = line.strip()
        
        if line.lower().startswith("missing topics"):
            continue

        if len(line) < 20:
            continue

        if any(x in line.lower() for x in ["question", "context", "instruction"]):
            continue

        facts.append(line)

    return facts


# ---------------- SIMILARITY ----------------
def is_similar(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.75


# ---------------- CLASSIFICATION ----------------
def classify_fact(fact):
    f = fact.lower()

    if "deadlock is" in f:
        return "definition"
    if "mutual exclusion" in f or "hold and wait" in f or "circular wait" in f:
        return "condition"
    if "due to" in f or "occurs" in f:
        return "cause"
    if "prevent" in f:
        return "prevention"
    if "example" in f:
        return "example"

    return "other"


# ---------------- MEMORY UPDATE ----------------
def update_memory(state, reasoning, confidence):

    # --------- BASIC FILTERS ----------
    if not reasoning or "error" in reasoning.lower():
        print("Skipping invalid reasoning")
        return

    if confidence <= 0.6:
        return

    if not hasattr(state, "covered_topics"):
        state.covered_topics = set()

    cleaned = clean_text(reasoning)
    facts = extract_knowledge(cleaned)

    if not facts:
        return

    # ---------------- STORE FACTS ----------------
    for fact in facts:

        fact_lower = fact.lower()

        # ❌ remove garbage
        if any(x in fact_lower for x in [
            "missing topics",
            "based on the given context",
            "i extracted",
            "not found in context"
        ]):
            continue

        if len(fact.split()) < 6:
            continue

        topic = classify_fact(fact)

        # ✅ allow ONE per topic
        existing_topics = [classify_fact(m["text"]) for m in state.memory]

        if topic in existing_topics:
            continue

        state.memory.append({
            "text": fact.strip(),
            "confidence": confidence
        })

        state.covered_topics.add(topic)

    # ---------------- LIMIT MEMORY (SIMPLE) ----------------
    # keep max 4 (definition + others)
    if len(state.memory) > 4:
        state.memory = state.memory[-4:]

    # ---------------- UPDATE COVERED TOPICS ----------------
    state.covered_topics = set(classify_fact(m["text"]) for m in state.memory)

    # ---------------- REBUILD CONTEXT ----------------
    state.context = " ".join(m["text"] for m in state.memory)