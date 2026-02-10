from typing import List, Optional, Tuple

import numpy as np
from toolkit import maths
from lapsim.encoder.encoder import extract_features
from lapsim.eval.evaluation import Evaluation
from toolkit.tracks.models import SegmentationLine, Track

"""Evaluation toolkit module.

This module provides the functions to evaluating two sets of spliced data. 
"""


def evaluate(truth: Track, predicted: Track) -> Evaluation:
    """Compare spliced data

    Args:
        truth: The ground truth data
        predicted: The data predicted by the model

    Returns:
        Evaluation model
    """
    _all_vels = [x.vel for x in truth.segmentations] + [x.vel for x in predicted.segmentations]
    min_vel, max_vel = min(_all_vels), max(_all_vels)

    true_optimal_line = calculate_optimal_positions(truth)
    estimated_optimal_line = calculate_optimal_positions(predicted)

    position_deltas, position_percentage_errors = [], []
    velocity_deltas, velocity_percentage_errors = [], []

    for n in range(len(truth.segmentations)):
        # Velocity errors
        vel_delta = truth.segmentations[n].vel - predicted.segmentations[n].vel
        velocity_deltas.append(vel_delta)
        velocity_percentage_errors.append(abs(vel_delta) / (max_vel - min_vel) * 100)

        # Position errors
        distance = maths.distance(true_optimal_line[n], estimated_optimal_line[n])
        if predicted.segmentations[n].pos < truth.segmentations[n].pos:
            distance *= -1
        position_deltas.append(distance)

        position_percentage_errors.append(abs(predicted.segmentations[n].pos - truth.segmentations[n].pos) * 100)

    apexes = find_apexes(truth.segmentations)

    return Evaluation.from_errors(
        laptime=estimate_lap_time(truth),
        predicted_laptime=estimate_lap_time(predicted),
        position_deltas=position_deltas,
        position_percentage_errors=position_percentage_errors,
        velocity_deltas=velocity_deltas,
        velocity_percentage_errors=velocity_percentage_errors,
        apexes=apexes
    )


def evaluate2(truth: Track, predicted: Track) -> Evaluation:
    """Compare spliced data irrespective of smoothing

    Args:
        truth: The ground truth data
        predicted: The data predicted by the model

    Returns:
        Evaluation model
    """
    _all_vels = [x.vel for x in truth.segmentations] + [x.vel for x in predicted.segmentations]
    min_vel, max_vel = min(_all_vels), max(_all_vels)

    velocity_deltas, velocity_percentage_errors = [], []

    for n in range(len(truth.segmentations)):
        # Velocity errors
        vel_delta = truth.segmentations[n].vel - predicted.segmentations[n].vel
        velocity_deltas.append(vel_delta)
        velocity_percentage_errors.append(abs(vel_delta) / (max_vel - min_vel) * 100)

    positional_deltas, percentage_deltas = evaluate_position_errors_irrespective_of_smoothing(truth, predicted)

    apexes = find_apexes(truth.segmentations)

    return Evaluation.from_errors(
        laptime=estimate_lap_time(truth),
        predicted_laptime=estimate_lap_time(predicted),
        position_deltas=positional_deltas,
        position_percentage_errors=percentage_deltas,
        velocity_deltas=velocity_deltas,
        velocity_percentage_errors=velocity_percentage_errors,
        apexes=apexes
    )


def estimate_lap_time(track: Track) -> float:
    """ Estimate the lap-time from a set of segmentation lines.

    This function first computes the optimal line then uses
    suvat equations to estimate the lap-time.

    Args:
        track: Spliced data

    Returns:
        Estimation lap time.
    """
    path = calculate_optimal_positions(track)

    total_time = 0
    for i in range(len(track.segmentations)):
        p_pos, c_pos = path[i - 1], path[i]
        u = track.segmentations[i].vel
        v = track.segmentations[i - 1].vel
        s = maths.distance(p_pos, c_pos)
        t = 2 * s / (u + v)

        total_time += t

    return total_time


def find_apexes(segmentations: List[SegmentationLine]) -> List[int]:
    """This function is designed to find the apexes of a track. Returning a
    list of indexes where each index maps to a seg line where the line kisses
    the apex.

    Args:
        List of segmentation lines with the ground truth position.

    Returns:
        List of indexes representing the apexes of the track.
    """
    _, angles, _ = extract_features(segmentations)
    apexes = []
    for i, normal in enumerate(segmentations):
        threshold = 0.02

        within_threshold = normal.pos < threshold or normal.pos > 1 - threshold
        within_ratio = abs(angles[i]) > 0.01

        if within_threshold and within_ratio:
            apexes += [[i, normal.pos]]

    apex_groups = []
    current_group = []
    for i in range(len(apexes)):
        current_group += [apexes[i]]
        if apexes[i + 1 - len(apexes)][0] != apexes[i][0] + 1:
            apex_groups += [current_group]
            current_group = []

    apexes = []
    for group in apex_groups:
        errors = [x[1] if x[1] < 0.5 else 1 - x[1] for x in group]
        apexes.append(group[np.argmin(errors)][0])

    return apexes


def calculate_optimal_positions(track: Track) -> np.ndarray:
    """ Calculate optimal positions from the track.

    This function will interpolate the position from the p value on the line
    and return the optimal line as defined by the given segmentations.

    Args:
        segmentations: spliced data

    Returns:
        Calculated line.
    """
    return np.array([
        [
            line.x1 + (line.x2 - line.x1) * line.pos,
            line.y1 + (line.y2 - line.y1) * line.pos,
        ]
        for line in track.segmentations
    ])


def evaluate_position_errors_irrespective_of_smoothing(truth: Track, predicted: Track) -> Optional[Tuple[List[float], List[float]]]:
    """Calculates the positional errors between the truth line and racing line irrespective of the sloping

    :param truth:
    :param predicted:
    :return:
    """
    # Create the track normals from the perspective of the line
    true_racing_line = calculate_optimal_positions(truth)
    racing_line_normals = maths.create_line_normals_from_points(true_racing_line, 10)

    # Spline the predicted line x5ing the points in a loop
    predicted_racing_line = maths.catmull_rom_spline(calculate_optimal_positions(predicted).tolist(), 5, True)

    absolute_errors = []
    percentage_errors = []
    for idx, normal in enumerate(racing_line_normals):
        error = None
        intersections = maths.segment_intersections(normal, maths.points_to_lines(predicted_racing_line))

        for intersection in intersections:
            delta = maths.distance(true_racing_line[idx], intersection)
            if error is None:
                error = delta
            else:
                error = min(error, delta)

        absolute_errors.append(error)
        percentage_errors.append(
            (error / truth.segmentations[idx].length)
            if error is not None else None
        )

    if not absolute_errors:
        return None

    return absolute_errors, percentage_errors
