import numpy as np
from typing import List, Tuple, Dict

def get_neighbors(r: int, c: int, H: int, W: int, connectivity: int = 4) -> List[Tuple[int, int]]:
    neighbors = []
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else: # 8-connectivity
        deltas = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            neighbors.append((nr, nc))
    return neighbors

def cca_recursive(grid: np.ndarray, connectivity: int = 4, background: int = 0) -> List[Dict]:
    """
    Performs Connected Component Analysis.
    Returns a list of objects, where each object is a mask (boolean grid) and color.
    """
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    objects = []

    for r in range(H):
        for c in range(W):
            color = grid[r, c]
            if color == background or visited[r, c]:
                continue
            
            # Start new component
            mask = np.zeros((H, W), dtype=bool)
            stack = [(r, c)]
            visited[r, c] = True
            mask[r, c] = True
            
            while stack:
                curr_r, curr_c = stack.pop()
                for nr, nc in get_neighbors(curr_r, curr_c, H, W, connectivity):
                    if not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        mask[nr, nc] = True
                        stack.append((nr, nc))
            
            objects.append({'mask': mask, 'color': int(color)})
            
    return objects

def cca_4connected(grid: np.ndarray) -> List[Dict]:
    return cca_recursive(grid, connectivity=4)

def cca_8connected(grid: np.ndarray) -> List[Dict]:
    return cca_recursive(grid, connectivity=8)

def cca_color_agnostic(grid: np.ndarray) -> List[Dict]:
    """
    Treats all non-background pixels as the same 'color' for connectivity,
    but preserves original colors in the object definition if needed (or just treats as one blob).
    Here we return the blob with a placeholder color or dominant color.
    """
    H, W = grid.shape
    binary_grid = (grid != 0).astype(int)
    # Use a temporary grid where all non-bg are 1
    # We pass this to cca_recursive, but we need to handle the fact that '1' is the color.
    
    # Actually, let's just reimplement/wrap logic for color agnostic
    visited = np.zeros((H, W), dtype=bool)
    objects = []
    
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0 or visited[r, c]:
                continue
                
            mask = np.zeros((H, W), dtype=bool)
            stack = [(r, c)]
            visited[r, c] = True
            mask[r, c] = True
            
            while stack:
                curr_r, curr_c = stack.pop()
                for nr, nc in get_neighbors(curr_r, curr_c, H, W, connectivity=4):
                    if not visited[nr, nc] and grid[nr, nc] != 0: # Connect to any non-background
                        visited[nr, nc] = True
                        mask[nr, nc] = True
                        stack.append((nr, nc))
            
            # For color, we can use the mode of the object pixels
            object_pixels = grid[mask]
            if len(object_pixels) > 0:
                vals, counts = np.unique(object_pixels, return_counts=True)
                dominant_color = vals[np.argmax(counts)]
            else:
                dominant_color = 1 # Fallback
                
            objects.append({'mask': mask, 'color': int(dominant_color)})
            
    return objects

def rectangular_cover(grid: np.ndarray) -> List[Dict]:
    """
    Decomposes the grid into rectangles.
    Simple greedy approach: find top-left unvisited non-bg pixel, 
    expand max rectangle of same color.
    """
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    objects = []
    
    for r in range(H):
        for c in range(W):
            color = grid[r, c]
            if color == 0 or visited[r, c]:
                continue
            
            # Expand width
            w = 0
            while c + w < W and grid[r, c + w] == color and not visited[r, c + w]:
                w += 1
            
            # Expand height
            h = 0
            while r + h < H:
                # Check if row segment is valid
                valid_row = True
                for k in range(w):
                    if grid[r + h, c + k] != color or visited[r + h, c + k]:
                        valid_row = False
                        break
                if not valid_row:
                    break
                h += 1
            
            # Mark visited and create mask
            mask = np.zeros((H, W), dtype=bool)
            mask[r:r+h, c:c+w] = True
            visited[r:r+h, c:c+w] = True
            
            objects.append({'mask': mask, 'color': int(color)})
            
    return objects
