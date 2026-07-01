import numpy as np
from typing import List, Tuple, Optional

def create_line_normals_from_points(points: List[Tuple[float, float]], length: float = 10.0) -> np.ndarray:
    """Create normal lines from a list of points

    Args:
        points: List of points to form normals from
        length: Length of the line

    Returns:
        Normal lines as an Nx4 array
    """
    pts = np.asarray(points)
    count = len(pts)
    if count < 3:
        return np.zeros((0, 4))

    # Get prev, current, next points
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)

    # Calculate angles to prev and next points
    # to_angle = angle_to(point, prev_point) = atan2(prev.y - pt.y, prev.x - pt.x)
    to_angle = np.arctan2(prev_pts[:, 1] - pts[:, 1], prev_pts[:, 0] - pts[:, 0])
    
    # angle3(prev, point, next) = (atan2(next.y - pt.y, next.x - pt.x) - atan2(prev.y - pt.y, prev.x - pt.x) + 2*pi) % (2*pi) - pi
    angle_next = np.arctan2(next_pts[:, 1] - pts[:, 1], next_pts[:, 0] - pts[:, 0])
    current_angle = (angle_next - to_angle + 2 * np.pi) % (2 * np.pi) - np.pi
    
    # normal = to_angle + (current_angle / 2)
    normal_angles = to_angle + (current_angle / 2.0)
    
    half_width = length / 2.0
    
    # Rotation:
    # left_end = rotate((point[0], point[1] - half_width), normal, point)
    # right_end = rotate((point[0], point[1] + half_width), normal, point)
    # rotate((px, py), angle, (cx, cy)):
    #   dx = px - cx, dy = py - cy
    #   nx = cos(angle) * dx - sin(angle) * dy + cx
    #   ny = sin(angle) * dx + cos(angle) * dy + cy
    
    # For left_end: dx = 0, dy = -half_width
    # lx = cos(normal) * (0) - sin(normal) * (-half_width) + pt.x = sin(normal) * half_width + pt.x
    # ly = sin(normal) * (0) + cos(normal) * (-half_width) + pt.y = -cos(normal) * half_width + pt.y
    
    # For right_end: dx = 0, dy = half_width
    # rx = cos(normal) * (0) - sin(normal) * (half_width) + pt.x = -sin(normal) * half_width + pt.x
    # ry = sin(normal) * (0) + cos(normal) * (half_width) + pt.y = cos(normal) * half_width + pt.y
    
    sin_n = np.sin(normal_angles)
    cos_n = np.cos(normal_angles)
    
    lx = sin_n * half_width + pts[:, 0]
    ly = -cos_n * half_width + pts[:, 1]
    rx = -sin_n * half_width + pts[:, 0]
    ry = cos_n * half_width + pts[:, 1]
    
    return np.stack([lx, ly, rx, ry], axis=1).tolist()

def create_normals_on_path(path: List[Tuple[float, float]], width: float, spacing: float) -> List[Tuple[float, float, float, float]]:
    """Create normals on path

    Args:
        path: The points to calculate normals using
        width: The width of the lines
        spacing: The gap between lines

    Returns:
        The generated lines as an Nx4 array
    """
    from .points import get_points_on_paths
    spaced_points = get_points_on_paths(path, spacing, loop=True)
    return create_line_normals_from_points(spaced_points, width)

def trim_normals_to_boundary(lines: List[Tuple[float, float, float, float]], left_boundary: List[Tuple[float, float]], right_boundary: List[Tuple[float, float]]) -> List[Tuple[float, float, float, float]]:
    """Trim normals to boundary
    
    Args:
        lines: List of lines (x1, y1, x2, y2)
        left_boundary: List of points for left boundary
        right_boundary: List of points for right boundary
        
    Returns:
        Trimmed lines as a list of tuples
    """
    from .c.points import points_to_lines, closest_point
    from .c.intersections import segment_intersections
    
    lines_arr = np.asarray(lines)
    n_lines = len(lines_arr)
    
    centers_x = (lines_arr[:, 0] + lines_arr[:, 2]) / 2.0
    centers_y = (lines_arr[:, 1] + lines_arr[:, 3]) / 2.0
    
    lb = points_to_lines(left_boundary)
    rb = points_to_lines(right_boundary)
    
    new_normals = []
    
    for i in range(n_lines):
        line = lines_arr[i]
        cx, cy = centers_x[i], centers_y[i]
        
        left_line = (line[0], line[1], cx, cy)
        right_line = (line[2], line[3], cx, cy)
        
        left_inters = segment_intersections(left_line, lb)
        right_inters = segment_intersections(right_line, rb)
        
        l_int = closest_point((cx, cy), left_inters)
        r_int = closest_point((cx, cy), right_inters)
        
        if l_int is None:
            l_int = tuple(line[0:2])
        if r_int is None:
            r_int = tuple(line[2:4])
            
        new_normals.append(tuple(l_int) + tuple(r_int))
        
    return new_normals
