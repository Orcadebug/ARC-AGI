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

    # 8. Border (Simplified check for solid border)
    # Check if first/last row and col are same non-zero color
    if H > 2 and W > 2:
        top = grid[0, :]
        bottom = grid[-1, :]
        left = grid[:, 0]
        right = grid[:, -1]
        # Check if all border pixels are same color and not 0
        border_pixels = np.concatenate([top, bottom, left, right])
        unique_border = np.unique(border_pixels)
        has_border = 1.0 if len(unique_border) == 1 and unique_border[0] != 0 else 0.0
    else:
        has_border = 0.0

    # 9. Background Color (Most frequent color, usually 0 but can be others)
    # We'll return the color index directly
    counts = np.bincount(grid.flatten())
    background_color = np.argmax(counts)

    # 10. Unique Object Count (already have num_objects, maybe this means unique shapes?)
    # Let's assume unique shapes based on shape_hash from object features
    # We need to extract object features first to do this properly, but here we just have the list of objects
    # Let's assume 'objects' list has 'mask'
    # We can compute shape hash locally or assume it's passed.
    # For now, let's just use number of unique colors as a proxy for unique objects if we define object by color
    # Or better, let's count unique masks (expensive).
    # Let's stick to unique colors for now as a simple proxy or just reuse num_objects if that's what it meant.
    # Requirement says "unique_object". Let's assume it means "count of unique object SHAPES".
    # We will implement a simple shape signature check.
    unique_shapes = set()
    for obj in objects:
        # Simple shape signature: (height, width, area)
        r, c = np.where(obj['mask'])
        if len(r) > 0:
            h = np.max(r) - np.min(r) + 1
            w = np.max(c) - np.min(c) + 1
            area = len(r)
            unique_shapes.add((h, w, area))
    num_unique_shapes = len(unique_shapes)

    # 11. Majority Color (excluding background if background is 0, or just most frequent non-bg)
    # If background is the most frequent, then majority_color is the 2nd most frequent
    # If background is not 0, it's just the most frequent.
    # Let's define majority_color as the most frequent NON-background color.
    counts_no_bg = counts.copy()
    if background_color == 0:
        counts_no_bg[0] = 0
    else:
        # If background is not 0, we might still want the most frequent color that is NOT the background
        counts_no_bg[background_color] = 0
    
    if np.sum(counts_no_bg) > 0:
        majority_color = np.argmax(counts_no_bg)
    else:
        majority_color = 0 # No other colors

    features = np.array([
        grid_width,
        grid_height,
        num_objects,
        num_colors,
        has_symmetry_x,
        has_symmetry_y,
        has_symmetry_diag,
        density,
        has_border,
        float(background_color),
        float(num_unique_shapes),
        float(majority_color)
    ], dtype=np.float32)
    
    return features
