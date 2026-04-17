# core/controller.py

def decide_action(state):

    if state.confidence > 0.85:
        return "STOP"

    if state.steps >= 6:
        return "STOP"

    # If last action was RETRIEVE → now REASON
    if state.last_action == "RETRIEVE":
        return "REASON"

    # If no context → retrieve
    if not state.context:
        return "RETRIEVE"

    # If low confidence → retrieve
    if state.confidence < 0.5:
        return "RETRIEVE"
    
    if state.last_action == "REASON" and state.confidence < 0.6:
        return "RETRIEVE"

    return "REASON"