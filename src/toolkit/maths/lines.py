import numpy as np
from typing import List, Tuple, Union, Optional

def line_centers(lines: List[Tuple[float, float, float, float]]) -> List[List[float]]:
    """Calculate the line centers from a list of lines."""
    l_arr = np.asarray(lines)
    if l_arr.size == 0:
        return []
    centers = (l_arr[:, 0:2] + l_arr[:, 2:4]) / 2.0
    return centers.tolist()

def line_length(line: Tuple[float, float, float, float]) -> float:
    return float(np.hypot(line[0] - line[2], line[1] - line[3]))

def line_lengths(lines: np.ndarray) -> np.ndarray:
    """Calculate the length of each item in the list of lines"""
    l_arr = np.asarray(lines)
    if l_arr.size == 0:
        return []
    lengths = np.hypot(l_arr[:, 0] - l_arr[:, 2], l_arr[:, 1] - l_arr[:, 3])
    return lengths

def normalise_lines(lines: List[Tuple[float, float, float, float]]) -> List[Optional[Tuple[float, float]]]:
    """Normalise a list of lines to a list of points"""
    l_arr = np.asarray(lines)
    if l_arr.size == 0:
        return []
    
    diffs = l_arr[:, 2:4] - l_arr[:, 0:2]
    lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    
    mask = lengths != 0
    res = [None] * len(lines)
    
    norms = diffs[mask] / lengths[mask][:, np.newaxis]
    
    # Fill the list
    norm_idx = 0
    for i in range(len(lines)):
        if mask[i]:
            res[i] = tuple(norms[norm_idx])
            norm_idx += 1
            
    return res

def set_line_lengths(normals: List[Tuple[float, float, float, float]], widths: List[float]) -> List[List[float]]:
    """Set the lengths of the given lines."""
    norm_pts = normalise_lines(normals)
    old_widths = line_lengths(normals)
    
    n_arr = np.asarray(normals)
    w_arr = np.asarray(widths)
    ow_arr = np.asarray(old_widths)
    
    # We need norm_pts as an array, handling None
    # Assuming normals with 0 length are rare or should be handled
    # If a line has length 0, normalise_lines returns None.
    
    res = []
    for i in range(len(normals)):
        n_pt = norm_pts[i]
        if n_pt is None:
            res.append(list(normals[i]))
            continue
            
        offset_val = (ow_arr[i] - w_arr[i]) / 2.0
        offset_x = n_pt[0] * offset_val
        offset_y = n_pt[1] * offset_val
        
        res.append([
            n_arr[i, 0] + offset_x,
            n_arr[i, 1] + offset_y,
            n_arr[i, 2] - offset_x,
            n_arr[i, 3] - offset_y
        ])
    return res

def extend_lines(lines: List[Tuple[float, float, float, float]], amount: float = 20.0, min_width: float = 0.0) -> List[Optional[List[float]]]:
    """Extend lines by a given amount."""
    l_arr = np.asarray(lines)
    if l_arr.size == 0:
        return []
    
    lengths = np.hypot(l_arr[:, 0] - l_arr[:, 2], l_arr[:, 1] - l_arr[:, 3])
    amount2 = amount + amount
    
    res = [None] * len(lines)
    for i in range(len(lines)):
        l = lengths[i]
        if l != 0:
            # Original logic: delta = (l - max(l - amount2, min_width)) / 2
            # Wait, original logic: shortened_lines[i] = [x1, y1, x2, y2]
            # nx = (x2 - x1) * delta / l
            # x1 -= nx, y1 -= ny, x2 += nx, y2 += ny
            # This EXTENDS the line if amount is positive.
            # If amount=20, amount2=40. If l=100, max(100-40, 0)=60. delta = (100-60)/2 = 20.
            # delta/l = 20/100 = 0.2.
            # x1 -= 0.2*(x2-x1), x2 += 0.2*(x2-x1). Total length becomes 100 + 40 = 140. Correct.
            
            # Wait, delta = (l - max(l - amount2, min_width)) / 2
            # If amount is positive, l - amount2 is smaller than l.
            # max(l - amount2, min_width) will be between min_width and l (if l > min_width).
            # delta will be (l - something smaller)/2, which is positive.
            # Then it subtracts nx from x1 and adds to x2, extending it.
            
            d_val = (l - max(l - amount2, min_width)) / 2.0
            d_ratio = d_val / l
            
            dx = (l_arr[i, 2] - l_arr[i, 0]) * d_ratio
            dy = (l_arr[i, 3] - l_arr[i, 1]) * d_ratio
            
            res[i] = [
                l_arr[i, 0] - dx,
                l_arr[i, 1] - dy,
                l_arr[i, 2] + dx,
                l_arr[i, 3] + dy
            ]
    return res

def start_points(lines: List[Tuple[float, float, float, float]]) -> List[List[float]]:
    """Get the start points from a list of lines"""
    return [list(line[:2]) for line in lines]

def end_points(lines: List[Tuple[float, float, float, float]]) -> List[List[float]]:
    """Get the end points from a list of lines"""
    return [list(line[2:4]) for line in lines]

def lerp_points_on_lines(lines: List[Tuple[float, float, float, float]], interpolations: List[float]) -> List[Tuple[float, float]]:
    """Linearly interpolate points upon a list of lines."""
    l_arr = np.asarray(lines)
    i_arr = np.asarray(interpolations)
    if l_arr.size == 0:
        return []
    
    # x1 + (x2 - x1) * interp
    res_x = l_arr[:, 0] + (l_arr[:, 2] - l_arr[:, 0]) * i_arr
    res_y = l_arr[:, 1] + (l_arr[:, 3] - l_arr[:, 1]) * i_arr
    
    return [tuple(x) for x in np.stack([res_x, res_y], axis=1)]
