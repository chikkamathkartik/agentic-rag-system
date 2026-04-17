from core.controller import decide_action
from core.budget import check_budget
from retrieval.retriever import retrieve
from reasoning.generator import generate
from reasoning.verifier import verify
from memory.store import update_memory
from retrieval.critic import evaluate_retrieval


# ---------------- REDUNDANCY CHECK ----------------
def is_redundant(reasoning, memory):
    reasoning = reasoning.lower()

    for m in memory:
        mem_text = m["text"].lower()

        common_words = set(reasoning.split()) & set(mem_text.split())

        if len(common_words) > 5:
            return True

    return False


def run_agent(state):

    # ---------------- INIT ----------------
    state.stagnation_count = 0
    state.prev_context = ""

    if not hasattr(state, "retrieval_fail_count"):
        state.retrieval_fail_count = 0

    if not hasattr(state, "no_progress_steps"):
        state.no_progress_steps = 0

    MAX_STEPS = 6

    # ---------------- LOOP ----------------
    while True:

        if state.steps >= MAX_STEPS:
            print("Max steps reached → stopping")
            break

        if not check_budget(state):
            print("Budget exceeded")
            break

        action = decide_action(state)
        state.last_action = action

        state.history.append({
            "step": state.steps,
            "memory_size": len(state.memory),
            "confidence": state.confidence
        })

        print(f"\n[Step {state.steps}] Action: {action}")

        # ---------------- STOP ----------------
        if action == "STOP":
            break

        # ---------------- RETRIEVE ----------------
        elif action == "RETRIEVE":

            docs = retrieve(state.query)
            new_context = " ".join(docs)

            verdict = evaluate_retrieval(state.query, new_context)
            print(f"Retrieval Critic: {verdict}")

            if verdict == "BAD":
                state.retrieval_fail_count += 1

                if state.retrieval_fail_count >= 2:
                    print("Forcing reasoning despite bad retrieval")
                    state.context = new_context
                    state.last_action = "REASON"
                    continue

                state.query += " detailed explanation"
                continue

            elif verdict == "PARTIAL":
                state.context = new_context

            else:  # GOOD
                state.retrieval_fail_count = 0
                state.context = new_context

        # ---------------- REASON ----------------
        elif action == "REASON":

            reasoning = generate(state)
            print("RAW LLM OUTPUT:", reasoning)

            if not reasoning:
                state.last_action = "RETRIEVE"
                continue

            score = verify(state, reasoning)

            # ✅ SUCCESS
            if score > 0.5:
                print(f"✔ Verified (score={score})")

                prev_memory_size = len(state.memory)
                update_memory(state, reasoning, score)

                # ---------------- NEW KNOWLEDGE ----------------
                if len(state.memory) > prev_memory_size:
                    state.confidence = min(1.0, state.confidence + 0.3)
                    state.no_progress_steps = 0

                # ---------------- NO REAL PROGRESS ----------------
                else:
                    print("No improvement detected")

                    if is_redundant(reasoning, state.memory):
                        state.no_progress_steps += 1
                    else:
                        state.no_progress_steps = 0

                    print(f"No progress count: {state.no_progress_steps}")

                    if state.no_progress_steps >= 2:
                        print("No real progress → stopping")
                        break

                    state.query += " explanation example"
                    state.last_action = "RETRIEVE"
                    continue

            # ❌ FAILURE
            else:
                print(f"✘ Rejected (score={score})")
                state.confidence = max(0.2, state.confidence - 0.1)
                state.last_action = "RETRIEVE"
                continue

        # ---------------- STAGNATION ----------------
        if state.context == state.prev_context:
            state.stagnation_count += 1
        else:
            state.stagnation_count = 0

        if state.stagnation_count >= 2:
            print("No progress → stopping")
            break

        state.prev_context = state.context

        # ---------------- DEBUG ----------------
        print("Memory:", [m["text"] for m in state.memory])
        print("Confidence:", round(state.confidence, 2))

        state.steps += 1

    return state.memory