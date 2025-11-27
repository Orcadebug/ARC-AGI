import numpy as np
from typing import List, Dict
from ..dsl.grammar import Program, SubroutineCall
from ..detection.features import extract_object_features
from ..detection.cca import cca_4connected

def grid_similarity(output: np.ndarray, target: np.ndarray) -> float:
    return np.mean(output == target)

def object_matching_score(output_objects: List[Dict], target_objects: List[Dict]) -> float:
    # Simple IoU or centroid matching
    # This is a placeholder for a more complex matching algorithm
    # Returns 0.0 to 1.0
    if not target_objects:
        return 1.0 if not output_objects else 0.0
    if not output_objects:
        return 0.0
        
    # Simplified: just compare counts and average area difference
    count_score = 1.0 - abs(len(output_objects) - len(target_objects)) / max(len(target_objects), 1)
    return max(0, count_score)

def compute_fitness(program: Program, output_grid: np.ndarray, target_grid: np.ndarray) -> float:
    # Primary: Grid similarity
    similarity = grid_similarity(output_grid, target_grid)
    
    # Secondary: Object matching
    # output_objects = cca_4connected(output_grid)
    # target_objects = cca_4connected(target_grid)
    # obj_score = object_matching_score(output_objects, target_objects)
    
    # Penalty: Program length
    length_penalty = 0.02 * len(program.statements)
    
    # Penalty: Complexity
    # complexity_penalty = 0.01 * sum(primitive_complexity(p) for p in program)
    
    # Combined fitness
    # fitness = (0.7 * similarity + 0.3 * obj_score - length_penalty)
    
    # Simplified for now:
    fitness = similarity - length_penalty
    
    return max(0.0, fitness)
