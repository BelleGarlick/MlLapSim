import numpy as np
from typing import List, Tuple, Union

def segment_intersections(line: np.ndarray, segments: np.ndarray, return_indexes: bool = False):
    """Calculate points of intersections between one line and a list of other lines."""
    l_arr = np.asarray(line)
    s_arr = np.asarray(segments)
    
    if s_arr.size == 0:
        if return_indexes:
            return [], []
        return []

    x1, y1, x2, y2 = l_arr
    x3, y3, x4, y4 = s_arr[:, 0], s_arr[:, 1], s_arr[:, 2], s_arr[:, 3]
    
    # Line 1: a1*x + b1*y = c1
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    
    # Lines 2: a2*x + b2*y = c2
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    
    delta = a1 * b2 - a2 * b1
    
    # Avoid division by zero
    valid_delta = delta != 0
    
    # Bounding boxes for segment 1
    min_x_l = min(x1, x2)
    max_x_l = max(x1, x2)
    min_y_l = min(y1, y2)
    max_y_l = max(y1, y2)
    
    # Bounding boxes for segment 2
    min_x_s = np.minimum(x3, x4)
    max_x_s = np.maximum(x3, x4)
    min_y_s = np.minimum(y3, y4)
    max_y_s = np.maximum(y3, y4)
    
    # Combined bounding boxes (intersection of boxes)
    min_x = np.maximum(min_x_l, min_x_s) - 1e-9
    max_x = np.minimum(max_x_l, max_x_s) + 1e-9
    min_y = np.maximum(min_y_l, min_y_s) - 1e-9
    max_y = np.minimum(max_y_l, max_y_s) + 1e-9
    
    # Initialize results
    intersections = []
    valid_indexes = []
    
    # Compute intersection points where delta != 0
    # Use masking to avoid division by zero
    x = np.zeros_like(delta)
    y = np.zeros_like(delta)
    
    x[valid_delta] = (b2[valid_delta] * c1 - b1 * c2[valid_delta]) / delta[valid_delta]
    y[valid_delta] = (a1 * c2[valid_delta] - a2[valid_delta] * c1) / delta[valid_delta]
    
    # Check if intersection point is within bounding boxes
    is_valid = (valid_delta & 
                (x >= min_x) & (x <= max_x) & 
                (y >= min_y) & (y <= max_y))
    
    valid_indices = np.where(is_valid)[0]
    for idx in valid_indices:
        intersections.append((float(x[idx]), float(y[idx])))
        valid_indexes.append(int(idx))
        
    if return_indexes:
        return intersections, valid_indexes
    return intersections
