import numpy as np
from typing import List, Dict
from .cca import cca_4connected, cca_8connected, cca_color_agnostic, rectangular_cover

def generate_parses(grid: np.ndarray) -> List[List[Dict]]:
    """
    Generates multiple object parse hypotheses for a given grid.
    Returns a list of parses, where each parse is a list of object dictionaries.
    """
    parses = []
    
    # Parse 1: Strict 4-connectivity
    parses.append(cca_4connected(grid))
    
    # Parse 2: 8-connectivity (diagonal adjacency)
    parses.append(cca_8connected(grid))
    
    # Parse 3: Color-agnostic (all non-background as one class)
    # Note: This might be too aggressive if the whole grid is connected, 
    # but useful for "object is the shape formed by all pixels"
    parses.append(cca_color_agnostic(grid))
    
    # Parse 4: Rectangular decomposition
    parses.append(rectangular_cover(grid))
    
    return parses
