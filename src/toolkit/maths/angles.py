import numpy as np
import math
from typing import Tuple, List, Optional, Union

def rotate(point: Tuple[float, float], angle: float, around: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """Rotate a point around another point.

    Args:
        point: The point to rotate
        angle: The angle to rotate the point around
        around: The point to rotate it around, or (0, 0) if not given

    Returns:
        The rotated point
    """
    around_x = around[0] if around else 0.0
    around_y = around[1] if around else 0.0

    cos_f = math.cos(angle)
    sin_f = math.sin(angle)
    dif_x = point[0] - around_x
    dif_y = point[1] - around_y

    nx = cos_f * dif_x - sin_f * dif_y + around_x
    ny = sin_f * dif_x + cos_f * dif_y + around_y
    return (nx, ny)

def angle_to(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])

def angle_between(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Calculate the angle from a -> b -> c.

    This function will return the angle from a to c passing through b
    and the value will be a value from -pi - pi. This means the
    direction is preserved.

    Args:
        a: A point in 2D euclidian space.
        b: A point in 2D euclidian space.
        c: A point in 2D euclidian space.

    Return:
        The angle between a -> b -> c
    """
    return ((math.atan2(c[1] - b[1], c[0] - b[0]) -
             math.atan2(a[1] - b[1], a[0] - b[0])) + math.tau) % math.tau

def angle3(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Calculate the delta angle between points. Between -pi and pi. The
    direction is preserved. But a straight line will result in angle 0
    because the delta angle is 0.

    Args:
        a: Point 1
        b: Point 2
        c: Point 3

    Returns:
        The angle change from a straight line.
    """
    return ((math.atan2(c[1] - b[1], c[0] - b[0]) -
             math.atan2(a[1] - b[1], a[0] - b[0])) +
            math.tau) % math.tau - math.pi

def line_angle(line: Tuple[float, float, float, float]) -> float:
    """Get the angle of the line relative to the standard coord axis

    Args:
        line: The line to calc the angle of.

    Returns:
        The angle of the line
    """
    return math.atan2(line[3] - line[1], line[2] - line[0])

def angle_between_lines(line1: Tuple[float, float, float, float], line2: Tuple[float, float, float, float]) -> float:
    """Calculate the angle between two lines, using dot product"""
    d1 = (line1[2] - line1[0], line1[3] - line1[1])
    d2 = (line2[2] - line2[0], line2[3] - line2[1])

    p = d1[0] * d2[0] + d1[1] * d2[1]
    n1 = math.sqrt(d1[0] * d1[0] + d1[1] * d1[1])
    n2 = math.sqrt(d2[0] * d2[0] + d2[1] * d2[1])

    if n1 * n2 == 0:
        return 0.0

    cos_val = p / (n1 * n2)
    # Clamp for floating point errors
    if cos_val > 1.0: cos_val = 1.0
    elif cos_val < -1.0: cos_val = -1.0

    if round(cos_val, 8) == 1.0:
        return 0.0

    return math.acos(cos_val)

def multi_angle_between_lines(lines1: List[Tuple[float, float, float, float]], lines2: List[Tuple[float, float, float, float]]) -> List[float]:
    """Calculate the angles between a list of lines"""
    l1 = np.asarray(lines1)
    l2 = np.asarray(lines2)
    
    d1 = l1[:, 2:4] - l1[:, 0:2]
    d2 = l2[:, 2:4] - l2[:, 0:2]
    
    dot = np.sum(d1 * d2, axis=1)
    n1 = np.linalg.norm(d1, axis=1)
    n2 = np.linalg.norm(d2, axis=1)
    
    denom = n1 * n2
    
    # Avoid division by zero
    mask = denom != 0
    cos_vals = np.ones(len(l1))
    cos_vals[mask] = dot[mask] / denom[mask]
    
    # Clamp
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    
    # Handle the round(p / (n1 * n2), 8) == 1 case from original
    # Although np.acos(1.0) is 0.0 anyway.
    
    angles = np.arccos(cos_vals)
    return angles.tolist()
