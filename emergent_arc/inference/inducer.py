import numpy as np
from typing import List, Tuple, Optional
from ..evolution.island import Island
from ..evolution.fitness import compute_fitness
from ..dsl.executor import execute_dsl
from ..dsl.grammar import Program
from ..memory.subroutines import SubroutineLibrary
from ..memory.veterans import VeteranPool

class OnlineProgramInducer:
    def __init__(self, evolver_factory, subroutine_library: SubroutineLibrary, veteran_pool: VeteranPool):
        self.evolver_factory = evolver_factory # Function to create islands/evolver
        self.subroutines = subroutine_library
        self.veterans = veteran_pool
        self.validation_tasks = [] # Store recent tasks for validation

    def solve_task(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray, timeout: float = 60.0) -> List[np.ndarray]:
        """
        Solves the task and returns top-2 candidate predictions.
        """
        import time
        start_time = time.time()
        
        # Phase 1: Evolve on first training pair
        input_grid, target_grid = train_pairs[0]
        
        island = self.evolver_factory()
        veteran_weights = self.veterans.sample(island.pop_size // 2)
        if veteran_weights:
            for i, weights in enumerate(veteran_weights):
                island.population[i] = weights
        
        # Evolve with budget
        max_generations = 100
        for gen in range(max_generations):
            if time.time() - start_time > timeout * 0.8: # Reserve time for validation
                break
                
            def fitness_fn(weights):
                # In real system: use self.decoder.decode_weights_to_program(weights)
                # output = execute_dsl(program, input_grid)
                # return compute_fitness(program, output, target_grid)
                return np.random.random() # Placeholder
            
            island.evaluate(fitness_fn)
            island.step()
            
        # Phase 2: Cross-validate & Select Top-2
        candidates = island.get_migrants(10)
        
        # Mock validation scoring
        scored_candidates = []
        for cand in candidates:
            # Decode and validate on all train pairs
            # score = validate(cand, train_pairs)
            score = np.random.random() # Placeholder
            scored_candidates.append((score, cand))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Return top 2 predictions
        top_2 = []
        for _, cand_weights in scored_candidates[:2]:
            # Predict on test input
            # program = self.decoder.decode_weights_to_program(cand_weights)
            # prediction = execute_dsl(program, test_input)
            prediction = test_input.copy() # Placeholder
            top_2.append(prediction)
            
        # Ensure we always return 2 candidates (duplicate if needed)
        while len(top_2) < 2:
            top_2.append(test_input.copy())
            
        return top_2

    def consolidate(self, program: Program):
        if not program:
            return
        # Extract and potentially promote fragments
        self.subroutines.extractor.record_solution(program)
        
        # Check for promotion
        # self.subroutines.maybe_promote(fragment, self.validation_tasks)
        
        # Save successful agent weights
        # self.veterans.add(weights)
        pass
