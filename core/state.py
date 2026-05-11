# core/state.py
import time

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
        
        # Timing tracking
        self.start_time = time.time()
        self.end_time = None
        self.execution_time = None
        self.retrieval_time = None
        self.generation_time = None
        
    def get_execution_time(self):
        """Get total execution time in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time