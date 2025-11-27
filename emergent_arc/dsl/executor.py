import numpy as np
from typing import List, Dict, Any
from copy import deepcopy
from .primitives import *
from .grammar import Program, Statement, SubroutineCall, HaltAction
from ..detection.cca import cca_4connected
from ..detection.features import extract_object_features

class ExecutionContext:
    def __init__(self, grid: np.ndarray, max_steps: int = 8):
        self.grid = grid.copy()
        self.objects = cca_4connected(self.grid) # Initial object parsing
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False

    def refresh_objects(self):
        # Re-parse objects after grid changes
        self.objects = cca_4connected(self.grid)

def execute_dsl(program: Program, input_grid: np.ndarray, max_steps: int = 8) -> np.ndarray:
    ctx = ExecutionContext(input_grid, max_steps)
    
    for stmt in program.statements:
        if ctx.current_step >= ctx.max_steps or ctx.done:
            break
            
        action = stmt.action
        execute_action(action, ctx)
        ctx.current_step += 1
        
    return ctx.grid

def execute_action(action: Action, ctx: ExecutionContext):
    if isinstance(action, HaltAction):
        ctx.done = True
        return

    if isinstance(action, Rotate):
        # Rotate object
        # 1. Extract object subgrid
        # 2. Rotate
        # 3. Clear old position
        # 4. Place new position (handling collisions/bounds)
        # Simplified: Just rotate the mask and pixels if possible
        pass # To be implemented fully

    elif isinstance(action, Gravity):
        apply_gravity(ctx, action.direction)
        ctx.refresh_objects()

    elif isinstance(action, Delete):
        if 0 <= action.object_id < len(ctx.objects):
            obj = ctx.objects[action.object_id]
            ctx.grid[obj['mask']] = 0 # Set to background
            ctx.refresh_objects()

    elif isinstance(action, Recolor):
        if 0 <= action.object_id < len(ctx.objects):
            obj = ctx.objects[action.object_id]
            ctx.grid[obj['mask']] = action.new_color
            # No need to refresh objects if just recoloring, usually
            
    elif isinstance(action, Foreach):
        # Filter objects
        # Apply action to each
        pass

    # ... Implement other primitives ...

def apply_gravity(ctx: ExecutionContext, direction: int):
    H, W = ctx.grid.shape
    new_grid = np.zeros_like(ctx.grid)
    
    # Simple gravity: move all non-bg pixels to the edge
    # This destroys object structure if not careful. 
    # Spec says "Collapse objects toward grid edge".
    # We should move *objects* or *pixels*? Usually pixels in ARC gravity.
    
    if direction == 1: # Down
        for c in range(W):
            col = ctx.grid[:, c]
            pixels = col[col != 0]
            new_col = np.zeros(H, dtype=int)
            if len(pixels) > 0:
                new_col[-len(pixels):] = pixels
            new_grid[:, c] = new_col
    elif direction == 0: # Up
        for c in range(W):
            col = ctx.grid[:, c]
            pixels = col[col != 0]
            new_col = np.zeros(H, dtype=int)
            if len(pixels) > 0:
                new_col[:len(pixels)] = pixels
            new_grid[:, c] = new_col
    elif direction == 2: # Left
        for r in range(H):
            row = ctx.grid[r, :]
            pixels = row[row != 0]
            new_row = np.zeros(W, dtype=int)
            if len(pixels) > 0:
                new_row[:len(pixels)] = pixels
            new_grid[r, :] = new_row
    elif direction == 3: # Right
        for r in range(H):
            row = ctx.grid[r, :]
            pixels = row[row != 0]
            new_row = np.zeros(W, dtype=int)
            if len(pixels) > 0:
                new_row[-len(pixels):] = pixels
            new_grid[r, :] = new_row
            
    ctx.grid = new_grid
