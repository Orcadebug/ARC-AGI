import numpy as np
from typing import List, Dict, Optional

class VeteranPool:
    def __init__(self, max_size: int = 500):
        self.weights = [] # List of dicts
        self.max_size = max_size

    def add(self, weights: np.ndarray, task_id: Optional[str] = None):
        entry = {
            'weights': weights.copy(),
            'task_id': task_id,
            'age': 0,
            'descendants_fitness': []
        }
        
        if len(self.weights) < self.max_size:
            self.weights.append(entry)
        else:
            # Replace oldest low-performing veteran
            # Simplified: just replace random or oldest
            self.weights.pop(0)
            self.weights.append(entry)

    def sample(self, n: int) -> List[np.ndarray]:
        """Sample veterans for seeding new population"""
        if not self.weights:
            return []
            
        if len(self.weights) < n:
            # Return all available, maybe duplicated
            indices = np.random.choice(len(self.weights), size=n, replace=True)
            return [self.weights[i]['weights'] for i in indices]
        
        # Weighted sampling based on fitness/success could go here
        indices = np.random.choice(len(self.weights), size=n, replace=False)
        return [self.weights[i]['weights'] for i in indices]

    def age_all(self):
        for v in self.weights:
            v['age'] += 1
