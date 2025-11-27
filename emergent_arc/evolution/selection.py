import numpy as np
import random
from typing import List, Tuple

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], k: int = 5) -> np.ndarray:
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitness_scores[i])
    return population[best_idx]

def elitism_selection(population: List[np.ndarray], fitness_scores: List[float], elite_count: int) -> List[np.ndarray]:
    indices = np.argsort(fitness_scores)[-elite_count:]
    return [population[i] for i in indices]
