import numpy as np
from typing import List, Tuple, Optional
from ..evolution.island import Island
from ..evolution.fitness import compute_fitness
from ..dsl.executor import execute_dsl
from ..dsl.grammar import Program
from ..memory.subroutines import SubroutineLibrary
from ..memory.veterans import VeteranPool
from .decoder import ProgramDecoder

class OnlineProgramInducer:
    def __init__(self, evolver_factory, subroutine_library: SubroutineLibrary, veteran_pool: VeteranPool):
        self.evolver_factory = evolver_factory # Function to create islands/evolver
        self.subroutines = subroutine_library
        self.veterans = veteran_pool
        self.validation_tasks = [] # Store recent tasks for validation
        self.decoder = ProgramDecoder(subroutine_library)

    def solve_task(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray, timeout: float = 60.0) -> List[np.ndarray]:
        """
        Solves the task and returns top-2 candidate predictions.
        """
        import time
        import jax
        import jax.numpy as jnp
        import jax.flatten_util
        from ..network.snn import SpikingPolicyNetwork
        from ..dsl.executor import execute_dsl
        from ..evolution.fitness import compute_fitness
        from ..detection.features import extract_global_features
        from ..detection.cca import cca_4connected
        
        start_time = time.time()
        
        # Initialize SNN structure for parameter reshaping
        # Assuming default dims for now, should match island creation
        snn = SpikingPolicyNetwork()
        dummy_key = jax.random.PRNGKey(0)
        dummy_params = snn.init_params(dummy_key)
        flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(dummy_params)
        
        # Phase 1: Evolve on first training pair
        input_grid, target_grid = train_pairs[0]
        
        # Extract features once
        # Note: extract_global_features returns numpy, convert to jax
        input_objects = cca_4connected(input_grid)
        features = extract_global_features(input_grid, input_objects)
        features_jnp = jnp.array(features)
        # Pad features to match SNN input dim (124)
        padded_features = jnp.pad(features_jnp, (0, 124 - len(features_jnp)))
        
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
                try:
                    # 1. Unflatten weights to SNN params
                    # Ensure weights are jnp array
                    weights_jnp = jnp.array(weights)
                    params = unflatten_fn(weights_jnp)
                    
                    # 2. Run SNN to get tokens
                    state = snn.init_state()
                    tokens = []
                    # Run for fixed number of steps (e.g. 20) to generate program sequence
                    # Autoregressive-ish: Input is static features for now
                    # In full version, input could be updated state
                    curr_state = state
                    for _ in range(20):
                        curr_state, spikes, logits = snn.forward(params, curr_state, padded_features)
                        # Argmax logits to get token
                        token = int(jnp.argmax(logits))
                        tokens.append(token)
                        if token == 99: # Halt
                            break
                    
                    # 3. Decode to Program
                    program = self.decoder.decode_sequence(tokens)
                    
                    # 4. Execute
                    output = execute_dsl(program, input_grid)
                    
                    # 5. Compute Fitness
                    return compute_fitness(program, output, target_grid)
                except Exception as e:
                    # Fallback for stability
                    return 0.0
            
            island.evaluate(fitness_fn)
            island.step()
            
        # Phase 2: Cross-validate & Select Top-2
        candidates = island.get_migrants(10)
        
        # Mock validation scoring
        scored_candidates = []
        for cand_weights in candidates:
            # Decode and validate on all train pairs
            try:
                weights_jnp = jnp.array(cand_weights)
                params = unflatten_fn(weights_jnp)
                state = snn.init_state()
                tokens = []
                curr_state = state
                for _ in range(20):
                    curr_state, spikes, logits = snn.forward(params, curr_state, padded_features)
                    token = int(jnp.argmax(logits))
                    tokens.append(token)
                    if token == 99: break
                
                program = self.decoder.decode_sequence(tokens)
                
                total_score = 0
                for t_in, t_out in train_pairs:
                    out = execute_dsl(program, t_in)
                    total_score += compute_fitness(program, out, t_out)
                
                avg_score = total_score / len(train_pairs)
                scored_candidates.append((avg_score, cand_weights))
            except:
                scored_candidates.append((0.0, cand_weights))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Return top 2 predictions
        top_2 = []
        for _, cand_weights in scored_candidates[:2]:
            # Predict on test input
            # Re-run SNN on test input features
            try:
                test_objects = cca_4connected(test_input)
                t_features = extract_global_features(test_input, test_objects)
                t_features_jnp = jnp.array(t_features)
                t_padded = jnp.pad(t_features_jnp, (0, 124 - len(t_features_jnp)))
                
                weights_jnp = jnp.array(cand_weights)
                params = unflatten_fn(weights_jnp)
                state = snn.init_state()
                tokens = []
                curr_state = state
                for _ in range(20):
                    curr_state, spikes, logits = snn.forward(params, curr_state, t_padded)
                    token = int(jnp.argmax(logits))
                    tokens.append(token)
                    if token == 99: break
                
                program = self.decoder.decode_sequence(tokens)
                prediction = execute_dsl(program, test_input)
                top_2.append(prediction)
            except:
                top_2.append(test_input.copy())
            
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
