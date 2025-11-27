import numpy as np
from typing import Dict, List

def extract_object_features(obj: Dict, grid_shape: tuple) -> np.ndarray:
    """
    Extracts 7 features for a single object:
    0: color (mode)
    1: area (pixel count)
    2: centroid_x
    3: centroid_y
    4: bbox_width
    5: bbox_height
    6: shape_hash (rotation-invariant contour signature - simplified here)
    """
    mask = obj['mask']
    color = obj['color']
    
    rows, cols = np.where(mask)
    
    if len(rows) == 0:
        return np.zeros(7)
    
    # 1. Area
    area = len(rows)
    
    # 2 & 3. Centroid
    centroid_y = np.mean(rows)
    centroid_x = np.mean(cols)
    
    # 4 & 5. Bounding Box
    min_r, max_r = np.min(rows), np.max(rows)
    min_c, max_c = np.min(cols), np.max(cols)
    bbox_height = max_r - min_r + 1
    bbox_width = max_c - min_c + 1
    
    # 6. Shape Hash (Simplified: aspect ratio + density hash)
    # A real shape hash would use contour moments or similar.
    # Here we use a simple proxy: (area / bbox_area) * 255
    bbox_area = bbox_width * bbox_height
    density = area / bbox_area if bbox_area > 0 else 0
    shape_hash = int(density * 255)
    
    features = np.array([
        color,
        area,
        centroid_x,
        centroid_y,
        bbox_width,
        bbox_height,
        shape_hash
    ], dtype=np.float32)
    
    return features

def extract_global_features(grid: np.ndarray, objects: List[Dict]) -> np.ndarray:
    """
    Extracts 8 global grid features:
    0: grid_width
    1: grid_height
    2: num_objects
    3: num_colors
    4: has_symmetry_x
    5: has_symmetry_y
    6: has_symmetry_diag
    7: density
    """
    H, W = grid.shape
    
    # 0 & 1. Dimensions
    grid_width = W
    grid_height = H
    
    # 2. Num Objects
    num_objects = len(objects)
    
    # 3. Num Colors
    unique_colors = np.unique(grid)
    num_colors = len(unique_colors[unique_colors != 0]) # Exclude background
    
    # 4, 5, 6. Symmetries
    # Check if grid equals its reflection
    # Horizontal symmetry (flip left-right) - actually usually means flip across Y axis (vertical axis)
    # But spec says "has_symmetry_x" and "has_symmetry_y".
    # Usually x-symmetry means symmetric along x-axis (flip up-down).
    # Let's assume:
    # symmetry_x: flip up-down (axis 0)
    # symmetry_y: flip left-right (axis 1)
    
    grid_flip_ud = np.flipud(grid)
    grid_flip_lr = np.fliplr(grid)
    grid_transpose = grid.T
    
    has_symmetry_x = 1.0 if np.array_equal(grid, grid_flip_ud) else 0.0
    has_symmetry_y = 1.0 if np.array_equal(grid, grid_flip_lr) else 0.0
    
    # Diagonal symmetry (only for square grids usually, but can check)
    if H == W:
        has_symmetry_diag = 1.0 if np.array_equal(grid, grid_transpose) else 0.0
    else:
        has_symmetry_diag = 0.0
        
    # 7. Density
    non_bg = np.sum(grid != 0)
    total_pixels = H * W
    density = non_bg / total_pixels if total_pixels > 0 else 0.0
    
    features = np.array([
        grid_width,
        grid_height,
        num_objects,
        num_colors,
        has_symmetry_x,
        has_symmetry_y,
        has_symmetry_diag,
        density
    ], dtype=np.float32)
    
    return features
