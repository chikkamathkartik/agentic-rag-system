# core/budget.py

def check_budget(state):
    if state.steps > 6:
        return False
    if state.tokens_used > 5000:
        return False
    return True