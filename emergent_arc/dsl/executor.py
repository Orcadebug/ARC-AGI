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
        if 0 <= action.object_id < len(ctx.objects):
            obj = ctx.objects[action.object_id]
            # Extract
            r, c = np.where(obj['mask'])
            if len(r) == 0: return
            min_r, max_r = np.min(r), np.max(r)
            min_c, max_c = np.min(c), np.max(c)
            h, w = max_r - min_r + 1, max_c - min_c + 1
            subgrid = ctx.grid[min_r:max_r+1, min_c:max_c+1].copy()
            # Mask out non-object pixels in subgrid (optional, but good for cleanliness)
            # For simplicity, assume rectangular object or just rotate the whole bounding box content
            # Better: use the mask to only rotate object pixels
            submask = obj['mask'][min_r:max_r+1, min_c:max_c+1]
            subgrid[~submask] = 0
            
            # Rotate
            k = action.degrees // 90
            rotated_subgrid = np.rot90(subgrid, k=k)
            
            # Clear old
            ctx.grid[obj['mask']] = 0
            
            # Place new (centered at same centroid if possible, or top-left aligned?)
            # Usually rotate around center.
            # Simplified: Place at same top-left for now, or center.
            # Let's try to keep centroid stable.
            new_h, new_w = rotated_subgrid.shape
            center_r, center_c = min_r + h//2, min_c + w//2
            new_top = center_r - new_h//2
            new_left = center_c - new_w//2
            
            # Bounds check
            H, W = ctx.grid.shape
            new_top = max(0, min(new_top, H - new_h))
            new_left = max(0, min(new_left, W - new_w))
            
            # Paste (handle collisions by overwriting)
            # Only paste non-zero pixels
            target_region = ctx.grid[new_top:new_top+new_h, new_left:new_left+new_w]
            mask_new = rotated_subgrid != 0
            target_region[mask_new] = rotated_subgrid[mask_new]
            ctx.grid[new_top:new_top+new_h, new_left:new_left+new_w] = target_region
            
            ctx.refresh_objects()

    elif isinstance(action, Flip):
        if 0 <= action.object_id < len(ctx.objects):
            obj = ctx.objects[action.object_id]
            r, c = np.where(obj['mask'])
            if len(r) == 0: return
            min_r, max_r = np.min(r), np.max(r)
            min_c, max_c = np.min(c), np.max(c)
            subgrid = ctx.grid[min_r:max_r+1, min_c:max_c+1].copy()
            submask = obj['mask'][min_r:max_r+1, min_c:max_c+1]
            subgrid[~submask] = 0
            
            if action.axis == 0: # Horizontal flip (across vertical axis) -> fliplr
                flipped_subgrid = np.fliplr(subgrid)
            else: # Vertical flip -> flipud
                flipped_subgrid = np.flipud(subgrid)
                
            ctx.grid[obj['mask']] = 0
            
            # Place back at same position
            target_region = ctx.grid[min_r:max_r+1, min_c:max_c+1]
            mask_new = flipped_subgrid != 0
            target_region[mask_new] = flipped_subgrid[mask_new]
            ctx.grid[min_r:max_r+1, min_c:max_c+1] = target_region
            
            ctx.refresh_objects()

    elif isinstance(action, Translate):
        if 0 <= action.object_id < len(ctx.objects):
            obj = ctx.objects[action.object_id]
            r, c = np.where(obj['mask'])
            if len(r) == 0: return
            
            # Clear old
            vals = ctx.grid[r, c]
            ctx.grid[r, c] = 0
            
            # Calculate new positions
            new_r = r + action.dy
            new_c = c + action.dx
            
            # Filter bounds
            H, W = ctx.grid.shape
            valid = (new_r >= 0) & (new_r < H) & (new_c >= 0) & (new_c < W)
            new_r = new_r[valid]
            new_c = new_c[valid]
            vals = vals[valid]
            
            # Place
            ctx.grid[new_r, new_c] = vals
            ctx.refresh_objects()

    elif isinstance(action, Gravity):
        apply_gravity(ctx, action.direction)
        ctx.refresh_objects()

    elif isinstance(action, Scale):
        # Simplified integer scaling (only 2x, 3x etc supported by factor)
        if 0 <= action.object_id < len(ctx.objects) and action.factor > 0:
            obj = ctx.objects[action.object_id]
            r, c = np.where(obj['mask'])
            if len(r) == 0: return
            min_r, max_r = np.min(r), np.max(r)
            min_c, max_c = np.min(c), np.max(c)
            subgrid = ctx.grid[min_r:max_r+1, min_c:max_c+1].copy()
            submask = obj['mask'][min_r:max_r+1, min_c:max_c+1]
            subgrid[~submask] = 0
            
            # Scale using Kronecker product for integer scaling
            scaled_subgrid = np.kron(subgrid, np.ones((action.factor, action.factor), dtype=int))
            
            ctx.grid[obj['mask']] = 0
            
            # Place (centered or top-left)
            # Just top-left for now
            H, W = ctx.grid.shape
            sh, sw = scaled_subgrid.shape
            
            # Clip if too big
            place_h = min(sh, H - min_r)
            place_w = min(sw, W - min_c)
            
            if place_h > 0 and place_w > 0:
                target_region = ctx.grid[min_r:min_r+place_h, min_c:min_c+place_w]
                source_region = scaled_subgrid[:place_h, :place_w]
                mask_new = source_region != 0
                target_region[mask_new] = source_region[mask_new]
                ctx.grid[min_r:min_r+place_h, min_c:min_c+place_w] = target_region
            
            ctx.refresh_objects()

    elif isinstance(action, Delete):
        if 0 <= action.object_id < len(ctx.objects):
            obj = ctx.objects[action.object_id]
            ctx.grid[obj['mask']] = 0 # Set to background
            ctx.refresh_objects()
            
    elif isinstance(action, Clone):
        if 0 <= action.object_id < len(ctx.objects):
            obj = ctx.objects[action.object_id]
            r, c = np.where(obj['mask'])
            if len(r) == 0: return
            
            vals = ctx.grid[r, c]
            
            new_r = r + action.offset_y
            new_c = c + action.offset_x
            
            H, W = ctx.grid.shape
            valid = (new_r >= 0) & (new_r < H) & (new_c >= 0) & (new_c < W)
            new_r = new_r[valid]
            new_c = new_c[valid]
            vals = vals[valid]
            
            ctx.grid[new_r, new_c] = vals
            ctx.refresh_objects()

    elif isinstance(action, Recolor):
        if 0 <= action.object_id < len(ctx.objects):
            obj = ctx.objects[action.object_id]
            ctx.grid[obj['mask']] = action.new_color
            
    elif isinstance(action, Foreach):
        # Simplified Foreach: Apply action to ALL objects
        # Ignoring Filter for now as it requires Predicate evaluation logic
        # And assuming action is a primitive that takes object_id
        # We will dynamically inject object_id
        
        # Copy list as it might change
        current_objects = list(range(len(ctx.objects)))
        
        for obj_id in current_objects:
            # Clone action and set object_id
            # This assumes action has 'object_id' field
            if hasattr(action.action, 'object_id'):
                new_action = deepcopy(action.action)
                new_action.object_id = obj_id
                execute_action(new_action, ctx)

    # ... Implement other primitives ...

def apply_gravity(ctx: ExecutionContext, direction: int):
    H, W = ctx.grid.shape
    new_grid = np.zeros_like(ctx.grid)
    
    # Simple gravity: move all non-bg pixels to the edge
    
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
