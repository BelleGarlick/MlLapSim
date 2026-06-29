import numpy as np
from typing import List, Tuple

def tj(ti: float, pi: Tuple[float, float], pj: Tuple[float, float], alpha: float) -> float:
    xi, yi = pi
    xj, yj = pj
    dx, dy = xj - xi, yj - yi
    l = (dx ** 2 + dy ** 2) ** 0.5
    return ti + l ** alpha

def sub_catmull_rom_spline(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    num_points: int,
    alpha: float = 0.5,
) -> List[Tuple[float, float]]:
    t0 = 0.0
    t1 = tj(t0, p0, p1, alpha)
    t2 = tj(t1, p1, p2, alpha)
    t3 = tj(t2, p2, p3, alpha)

    d0 = t1 - t0
    d1 = t2 - t1
    d2 = t3 - t2
    d3 = t2 - t0
    d4 = t3 - t1

    if d0 == 0 or d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0:
        # In case of overlapping points, we might have zero divisions.
        # Original code raises an exception.
        raise Exception("Invalid input")

    t_values = np.linspace(t1, t2, num_points + 1)
    
    items = []
    for t in t_values:
        e0 = (t1 - t) / d0
        e1a = (t2 - t) / d1
        e1b = (t2 - t) / d3
        e2a = (t3 - t) / d2
        e2b = (t3 - t) / d4
        e3a = (t - t0) / d0
        e3b = (t - t0) / d3
        e4a = (t - t1) / d1
        e4b = (t - t1) / d4
        e5 = (t - t2) / d2

        a1x = e0 * p0[0] + e3a * p1[0]
        a1y = e0 * p0[1] + e3a * p1[1]
        a2x = e1a * p1[0] + e4a * p2[0]
        a2y = e1a * p1[1] + e4a * p2[1]
        a3x = e2a * p2[0] + e5 * p3[0]
        a3y = e2a * p2[1] + e5 * p3[1]
        
        b1x = e1b * a1x + e3b * a2x
        b1y = e1b * a1y + e3b * a2y
        b2x = e2b * a2x + e4b * a3x
        b2y = e2b * a2y + e4b * a3y

        items.append((
            float(e1a * b1x + e4a * b2x),
            float(e1a * b1y + e4a * b2y)
        ))

    return items

def catmull_rom_spline(points: List[Tuple[float, float]], num_points: int = 10, loop: bool = False) -> List[Tuple[float, float]]:
    if len(points) == 0:
        return []

    if loop:
        control_points = [points[-1]] + points + points[:2]
    else:
        control_points = points

    all_splines = []
    # num_segments = len(control_points) - 3
    n = len(control_points)
    for i in range(n - 3):
        subspline = sub_catmull_rom_spline(
            control_points[i], 
            control_points[i+1], 
            control_points[i+2], 
            control_points[i+3], 
            num_points
        )
        all_splines += subspline[:-1]

    return all_splines
