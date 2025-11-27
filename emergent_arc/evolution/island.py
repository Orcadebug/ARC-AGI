import numpy as np
from typing import List, Tuple
from .pgpe import PGPEOptimizer
from .selection import tournament_selection, elitism_selection

class Island:
    def __init__(self, island_id: int, population_size: int, param_size: int):
        self.island_id = island_id
        self.pop_size = population_size
        self.optimizer = PGPEOptimizer(param_size)
        self.population = self.optimizer.ask(population_size)
        self.fitness_scores = [0.0] * population_size
        self.best_agent = None
        self.best_fitness = -1.0

    def evaluate(self, fitness_function):
        """Evaluate current population."""
        scores = []
        for agent in self.population:
            score = fitness_function(agent)
            scores.append(score)
        self.fitness_scores = scores
        
        # Track best
        max_idx = np.argmax(scores)
        if scores[max_idx] > self.best_fitness:
            self.best_fitness = scores[max_idx]
            self.best_agent = self.population[max_idx].copy()

    def step(self):
        """Perform one evolutionary step."""
        self.optimizer.tell(self.population, self.fitness_scores)
        self.population = self.optimizer.ask(self.pop_size)

    def get_migrants(self, count: int) -> List[np.ndarray]:
        """Return top 'count' agents for migration."""
        return elitism_selection(self.population, self.fitness_scores, count)

    def accept_migrants(self, migrants: List[np.ndarray]):
        """Replace worst agents with migrants."""
        # Sort by fitness
        indices = np.argsort(self.fitness_scores)
        # Replace worst 'len(migrants)'
        for i, migrant in enumerate(migrants):
            idx = indices[i]
            self.population[idx] = migrant
            # We should probably re-evaluate or just assume they are good, 
            # but for PGPE we update the distribution.
            # Actually PGPE 'ask' generates new pop from distribution.
            # Injecting migrants into PGPE is tricky.
            # Usually we just update the mean towards the migrants or add them to the sample set for 'tell'.
            # For simplicity in this hybrid approach:
            # We treat 'population' as the samples. If we replace samples, we affect the next 'tell'.
            pass
