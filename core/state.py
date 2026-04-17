# core/state.py

class State:
    def __init__(self, query):
        self.exploration_target = None
        self.query = query
        self.memory = []
        self.context = ""
        self.steps = 0
        self.tokens_used = 0
        self.confidence = 0.0
        self.last_action = None
        self.covered_topics = set()
        self.history = []
        self.covered_topics = set()