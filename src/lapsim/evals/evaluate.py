import numpy as np
from toolkit import maths
from lapsim.encoder.encoder import extract_features
# from lapsim.evals.evaluation import Evaluation
from toolkit.tracks.models import Track

"""Evaluation toolkit module.

This module provides the functions to evaluating two sets of spliced data. 
"""

# todo could look back into this in the future
# def evaluate2(truth: Track, predicted: Track) -> Evaluation:
#     """Compare spliced data irrespective of smoothing
#
#     Args:
#         truth: The ground truth data
#         predicted: The data predicted by the model
#
#     Returns:
#         Evaluation model
#     """
#     _all_vels = truth["vel"].tolist() + predicted["vel"].tolist()
#     min_vel, max_vel = min(_all_vels), max(_all_vels)
#
#     velocity_deltas, velocity_percentage_errors = [], []
#
#     for n in range(len(truth["track"])):
#         # Velocity errors
#         vel_delta = truth["vel"][n] - predicted["vel"][n]
#         velocity_deltas.append(vel_delta)
#         velocity_percentage_errors.append(abs(vel_delta) / (max_vel - min_vel) * 100)
#
#     positional_deltas, percentage_deltas = evaluate_position_errors_irrespective_of_smoothing(truth, predicted)
#
#     apexes = find_apexes(truth.segmentations)
#
#     return Evaluation.from_errors(
#         laptime=estimate_lap_time(truth),
#         predicted_laptime=estimate_lap_time(predicted),
#         position_deltas=positional_deltas,
#         position_percentage_errors=percentage_deltas,
#         velocity_deltas=velocity_deltas,
#         velocity_percentage_errors=velocity_percentage_errors,
#         apexes=apexes
#     )


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
    for i in range(len(track["track"])):
        p_pos, c_pos = path[i - 1], path[i]
        u = track["vel"][i]
        v = track["vel"][i - 1]
        s = maths.distance(p_pos, c_pos)
        t = 2 * s / (u + v)

        total_time += t

    return total_time


def find_apexes(track: dict) -> list[int]:
    """This function is designed to find the apexes of a track. Returning a
    list of indexes where each index maps to a seg line where the line kisses
    the apex.

    Args:
        List of segmentation lines with the ground truth position.

    Returns:
        List of indexes representing the apexes of the track.
    """
    _, angles, _ = extract_features(track["track"])
    apexes = []
    for i in range(len(angles)):
        threshold = 0.02

        within_threshold = track["pos"][i] < threshold or track["pos"][i] > 1 - threshold
        within_ratio = abs(angles[i]) > 0.01

        if within_threshold and within_ratio:
            apexes += [[i, track['pos'][i]]]

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


def calculate_optimal_positions(track: dict) -> np.ndarray:
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
            line[0] + (line[2] - line[0]) * pos,
            line[1] + (line[3] - line[1]) * pos,
        ]
        for (line, pos) in zip(track["track"], track["pos"])
    ])


def evaluate_position_errors_irrespective_of_smoothing(truth: Track, predicted: Track) -> (tuple[list[float], list[float]] | None):
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


def get_percentile(arr: Sequence[float], percentile: float) -> float:
    values = sorted(arr)
    idx = int(len(values) * percentile)
    return values[idx]


def create_eval_metrics_from_arr(prefix: str, arr):
    return {
        f"{prefix}mean": np.mean(arr),
        f"{prefix}mean-abs": np.mean(np.abs(arr)),
        f"{prefix}rmse": np.pow(np.mean(np.pow(arr, 2)), 1 / 2),
        f"{prefix}percentile-25": get_percentile(np.abs(arr), 0.25),
        f"{prefix}percentile-50": get_percentile(np.abs(arr), 0.50),
        f"{prefix}percentile-75": get_percentile(np.abs(arr), 0.75),
        f"{prefix}percentile-95": get_percentile(np.abs(arr), 0.95),
        f"{prefix}max": np.max(np.abs(arr)),
    }


def evaluate_laptimes(comparison_pairs, distance_per_seg_line=10):
    laptime_errors_labelled = [
        (
            calculated_track["id"],  # name
            (len(calculated_track["track"]) * distance_per_seg_line) / 1000,  # n kilometers
            estimate_lap_time(calculated_track) - estimate_lap_time(predicted_track))  # the error
        for calculated_track, predicted_track in comparison_pairs
    ]

    laptime_errors_normalised = [
        (key, error / length)
        for key, length, error in laptime_errors_labelled
    ]

    laptime_errors_values = [
        x[-1] for x in laptime_errors_labelled
    ]
    laptime_normalised_errors_values = [
        x[-1] for x in laptime_errors_normalised
    ]

    return {
        **create_eval_metrics_from_arr("laptime/", laptime_errors_values),
        "laptime/sorted": [list(x) for x in sorted(laptime_errors_labelled, key=lambda x: np.abs(x[-1]))],

        **create_eval_metrics_from_arr("laptime-per-kilometer/", laptime_normalised_errors_values),
        "laptime-normalised/sorted": [list(x) for x in sorted(laptime_errors_normalised, key=lambda x: np.abs(x[-1]))],
    }


def evaluate_positions(comparison_pairs):
    position_normalised_errors = [pair[0]["pos"] - pair[1]["pos"] for pair in comparison_pairs]
    error_signs = [np.sign(x) for x in position_normalised_errors]

    # Errors grouped by track
    track_errors = [
        np.linalg.norm(
            calculate_optimal_positions(pair[0]) \
            - calculate_optimal_positions(pair[1]),
            axis=1
        )
        for pair in comparison_pairs
    ]
    track_errors = [err * sign for err, sign in zip(track_errors, error_signs)]

    # map the apexes to the errors at those points
    track_apexes = [find_apexes(pair[0]) for pair in comparison_pairs]
    apex_errors = [errors[apexes] for errors, apexes in zip(track_errors, track_apexes)]

    all_normalised_errors = np.concatenate(position_normalised_errors)
    all_errors = np.concatenate(track_errors)
    all_apex_errors = np.concatenate(apex_errors)

    return {
        **create_eval_metrics_from_arr("position/", all_errors),
        **create_eval_metrics_from_arr("position/apex/", all_apex_errors),
        **create_eval_metrics_from_arr("position/normalised/", all_normalised_errors)
    }


def evaluate_velocities(comparison_pairs):
    velocity_errors = [pair[0]["vel"] - pair[1]["vel"] for pair in comparison_pairs]

    # map the apexes to the errors at those points
    track_apexes = [find_apexes(pair[0]) for pair in comparison_pairs]
    apex_errors = [errors[apexes] for errors, apexes in zip(velocity_errors, track_apexes)]

    all_velocity_errors = np.concatenate(velocity_errors)
    all_apex_errors = np.concatenate(apex_errors)

    return {
        **create_eval_metrics_from_arr("velocity/", all_velocity_errors),
        **create_eval_metrics_from_arr("velocity/apex/", all_apex_errors),
    }


def evaluate(pairs: list[tuple[dict, dict]], distance_per_seg_line=10) -> dict:
    return {
        **evaluate_laptimes(pairs, distance_per_seg_line=distance_per_seg_line),
        **evaluate_positions(pairs),
        **evaluate_velocities(pairs),
    }