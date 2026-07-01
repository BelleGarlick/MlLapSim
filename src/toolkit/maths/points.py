import numpy as np
from typing import List, Tuple, Union, Optional

def normalise_point(point: Tuple[float, float]) -> Tuple[float, float]:
    """Normalise a point"""
    x, y = point
    d = np.hypot(x, y)
    if d == 0:
        return (0.0, 0.0)
    return (float(x / d), float(y / d))

def normalise_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Normalise a list of points."""
    pts = np.asarray(points)
    if pts.size == 0:
        return []
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    mask = (norms != 0).flatten()
    res = np.zeros_like(pts, dtype=float)
    res[mask] = pts[mask] / norms[mask]
    return res

def distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """Calculate the distance between two points"""
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))

def distances(points: List[Tuple[float, float]], origin: Tuple[float, float]) -> List[float]:
    """Calculate the distances from the list of points to the origin."""
    pts = np.asarray(points)
    if pts.size == 0:
        return []
    org = np.asarray(origin)
    dists = np.linalg.norm(pts - org, axis=1)
    return dists.tolist()

def closest_point(origin: Tuple[float, float], points: List[Tuple[float, float]], return_index: bool = False) -> Union[Tuple[float, float], int, None]:
    """Find the closest point between the list of points and the given origin."""
    if not points:
        return None if not return_index else -1
    
    pts = np.asarray(points)
    org = np.asarray(origin)
    dists = np.linalg.norm(pts - org, axis=1)
    closest_idx = np.argmin(dists)
    
    if return_index:
        return int(closest_idx)
    return tuple(pts[closest_idx])

def points_to_lines(points: np.ndarray) -> np.ndarray:
    """Turn a list of points to lines."""
    count = len(points)
    if count == 0:
        return np.zeros((0, 4))

    next_pts = np.roll(points, -1, axis=0)
    
    return np.hstack([points, next_pts])

def sub_point(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])

def sub_points(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts_a = np.asarray(a)
    pts_b = np.asarray(b)
    res = pts_a - pts_b
    return [tuple(x) for x in res]

def interpolate_points_between(p1: Tuple[float, float], p2: Tuple[float, float], n: int) -> List[Tuple[float, float]]:
    """This function returns the interpolated points between two points."""
    if n <= 0:
        return []
    
    # Matching the original logic exactly to avoid precision issues
    x_start, y_start = p1
    x_range = p2[0] - x_start
    y_range = p2[1] - y_start
    
    interpolated_points = []
    for i in range(1, n + 1):
        portion = i / (n + 1)
        interpolated_points.append((
            x_start + (portion * x_range),
            y_start + (portion * y_range)
        ))
    return interpolated_points

def get_points_on_paths(path: List[Tuple[float, float]], spacing: float, loop: bool) -> List[Tuple[float, float]]:
    """Generate points on the given path"""
    from .lines import line_lengths
    lines = points_to_lines(path)
    if not loop: 
        lines = lines[:-1]
    
    lengths = np.asarray(line_lengths(lines))
    if len(lengths) == 0:
        return []
        
    cum_distances = np.zeros(len(lengths) + 1)
    cum_distances[1:] = np.cumsum(lengths)
    
    max_length = cum_distances[-1]
    if loop: 
        max_length -= spacing
    
    # Calculate cumulative spacings
    v = 0.0
    max_v = max_length + (spacing / 2.0)
    cumulative_spacings = []
    while v < max_v:
        cumulative_spacings.append(v)
        v += spacing
        
    if not cumulative_spacings:
        return []

    cumulative_spacings = np.array(cumulative_spacings)
    
    # Use searchsorted to find which line each spacing falls into
    # focus_idx = np.searchsorted(cum_distances, cumulative_spacings, side='right') - 1
    # But we need to handle edge cases where spacing is exactly at the end.
    focus_indices = np.searchsorted(cum_distances, cumulative_spacings, side='right') - 1
    # Clip to avoid out of bounds (though searchsorted with side='right' and cum_distances should be fine)
    focus_indices = np.clip(focus_indices, 0, len(lengths) - 1)
    
    portions = (cumulative_spacings - cum_distances[focus_indices]) / lengths[focus_indices]
    
    lines_arr = np.asarray(lines)
    focus_lines = lines_arr[focus_indices]
    
    start_pts = focus_lines[:, 0:2]
    end_pts = focus_lines[:, 2:4]
    
    res_pts = start_pts + portions[:, np.newaxis] * (end_pts - start_pts)
    return [tuple(x) for x in res_pts]

def lerp_point(point_a: Tuple[float, float], point_b: Tuple[float, float], portion: float) -> Tuple[float, float]:
    """Linearly interpolate a point between the two points"""
    pax, pay = point_a
    pbx, pby = point_b
    return (
        ((pbx - pax) * portion) + pax,
        ((pby - pay) * portion) + pay
    )

def add_points_lists(points_lists: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    if not points_lists:
        return []
    
    arr = np.asarray(points_lists)
    # arr shape: (n_lists, n_points, 2)
    summed = np.sum(arr, axis=0)
    return [tuple(x) for x in summed]
