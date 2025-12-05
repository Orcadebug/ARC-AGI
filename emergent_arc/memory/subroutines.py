from typing import List, Tuple, Dict, Counter
import numpy as np
from ..dsl.grammar import Program, Statement, Action

class FragmentExtractor:
    def __init__(self):
        self.action_traces = []
        self.fragment_counts = Counter()

    def record_solution(self, program: Program):
        """Log successful program traces"""
        # Convert program to tuple of actions (simplified representation)
        # We need a hashable representation of actions
        actions = tuple(str(stmt.action) for stmt in program.statements)
        self.action_traces.append(actions)
        
        # Extract all subsequences of length 2-4
        for length in range(2, 5):
            for i in range(len(actions) - length + 1):
                fragment = actions[i:i+length]
                self.fragment_counts[fragment] += 1

    def get_candidate_subroutines(self, min_frequency: float = 0.05) -> List[Tuple]:
        """Return fragments appearing in >5% of solutions"""
        if not self.action_traces:
            return []
            
        threshold = len(self.action_traces) * min_frequency
        candidates = [
            fragment for fragment, count in self.fragment_counts.items()
            if count >= threshold
        ]
        return sorted(candidates, key=lambda f: self.fragment_counts[f], reverse=True)

class SubroutineLibrary:
    def __init__(self, max_subroutines: int = 32):
        self.subroutines = [] # List of program fragments (tuples of actions)
        self.usage_counts = []
        self.max_size = max_subroutines
        self.extractor = FragmentExtractor()

    def maybe_promote(self, fragment: Tuple, validation_tasks: List):
        """Promote fragment to subroutine if it generalizes"""
        # Placeholder logic: In a real system, we'd test on validation tasks.
        # Here we just add it if space permits or replace least used.
        self.add_subroutine(fragment)
        return True

    def add_subroutine(self, fragment: Tuple):
        if len(self.subroutines) < self.max_size:
            self.subroutines.append(fragment)
            self.usage_counts.append(0)
        else:
            # Replace least-used subroutine
            if not self.usage_counts:
                return
            min_idx = np.argmin(self.usage_counts)
            self.subroutines[min_idx] = fragment
            self.usage_counts[min_idx] = 0

    def get_subroutine(self, index: int) -> Tuple:
        if 0 <= index < len(self.subroutines):
            return self.subroutines[index]
        return ()

    def get_available_subroutines(self) -> Dict[int, Tuple]:
        """Return a dictionary of available subroutine IDs and their signatures (actions)"""
        return {i: sub for i, sub in enumerate(self.subroutines)}
