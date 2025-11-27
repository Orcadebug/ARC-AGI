import numpy as np
from typing import List, Tuple

class PGPEOptimizer:
    def __init__(self, param_size: int, learning_rate: float = 0.01, sigma_init: float = 0.1):
        self.param_size = param_size
        self.lr = learning_rate
        self.sigma = np.full(param_size, sigma_init)
        self.mu = np.zeros(param_size)
        
        # Adam-like momentum
        self.m_mu = np.zeros(param_size)
        self.v_mu = np.zeros(param_size)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def ask(self, population_size: int) -> List[np.ndarray]:
        """Generate a population of parameters."""
        population = []
        for _ in range(population_size):
            # Sample from Gaussian N(mu, sigma)
            sample = self.mu + self.sigma * np.random.randn(self.param_size)
            population.append(sample)
        return population

    def tell(self, population: List[np.ndarray], fitness_scores: List[float]):
        """Update distribution parameters based on fitness scores."""
        self.t += 1
        
        pop_matrix = np.array(population)
        fitness_array = np.array(fitness_scores)
        
        # Rank-based fitness shaping (optional, but good for stability)
        # For now, just use raw fitness or normalized fitness
        # Let's use centered rank-based weights
        ranks = np.argsort(np.argsort(fitness_array)) # 0 to N-1
        N = len(population)
        # Linear ranking: -0.5 to 0.5
        weights = (ranks / (N - 1)) - 0.5
        
        # Estimate gradients
        # grad_mu = sum(w * (x - mu))
        # grad_sigma = sum(w * ((x - mu)^2 - sigma^2) / sigma)
        
        delta = pop_matrix - self.mu
        
        grad_mu = np.dot(weights, delta) / N
        
        # Update mu using Adam
        self.m_mu = self.beta1 * self.m_mu + (1 - self.beta1) * grad_mu
        self.v_mu = self.beta2 * self.v_mu + (1 - self.beta2) * (grad_mu ** 2)
        
        m_hat = self.m_mu / (1 - self.beta1 ** self.t)
        v_hat = self.v_mu / (1 - self.beta2 ** self.t)
        
        self.mu += self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Simple sigma decay or update
        # For simplicity, we can keep sigma constant or slowly decay
        # Or implement full PGPE sigma update
        # Let's just decay slightly to converge
        # self.sigma *= 0.999 
        pass
