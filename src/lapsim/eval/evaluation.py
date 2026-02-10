import dataclasses
from typing import List

import numpy as np


def _mean(x):
    non_null_x = [v for v in x if v is not None]
    if len(non_null_x) == 0:
        return None
    return sum(non_null_x) / len(non_null_x)


def _max(x):
    non_null_x = [v for v in x if v is not None]
    if len(non_null_x) == 0:
        return None
    return max(non_null_x)


# TODO Add tests and documentation
@dataclasses.dataclass
class EvaluationError:
    max: float
    mean: float
    mean_absolute: float
    rmse: float
    ci95: float

    percentage_mean: float
    percentage_max: float
    percentage_ci95: float

    apex_mean: float
    apex_mean_absolute: float
    apex_max: float

    @staticmethod
    def from_errors(deltas: List[float], percentage_errors: List[float], apexes: List[int]):
        deltas = np.array([d for d in deltas if d is not None])

        return EvaluationError(
            max=np.max(np.abs(deltas)),
            mean=float(np.mean(deltas)),
            mean_absolute=float(np.mean(np.abs(deltas))),
            rmse=np.sqrt(np.mean(np.square(deltas))),
            ci95=sorted(np.abs(deltas))[int(len(deltas) * 0.95)],

            percentage_mean=float(np.mean(percentage_errors)),
            percentage_max=np.max(percentage_errors),
            percentage_ci95=sorted(percentage_errors)[int(len(percentage_errors) * 0.95)],

            apex_mean=float(np.mean(deltas[apexes])) if apexes else None,
            apex_mean_absolute=float(np.mean(np.abs(deltas[apexes]))) if apexes else None,
            apex_max=np.max(np.abs(deltas[apexes])) if apexes else None,
        )

    # TODO Test
    @staticmethod
    def combine(comparisons: List['EvaluationError']) -> 'EvaluationError':
        return EvaluationError(
            max=max([item.max for item in comparisons]),
            mean=_mean([item.mean for item in comparisons]),
            mean_absolute=_mean([item.mean_absolute for item in comparisons]),
            rmse=_mean([item.rmse for item in comparisons]),
            ci95=_mean([item.ci95 for item in comparisons]),
            percentage_mean=_mean([item.percentage_mean for item in comparisons]),
            percentage_max=max([item.percentage_max for item in comparisons]),
            percentage_ci95=_mean([item.percentage_ci95 for item in comparisons]),
            apex_mean=_mean([item.apex_mean for item in comparisons]),
            apex_mean_absolute=_mean([item.apex_mean_absolute for item in comparisons]),
            apex_max=_max([item.apex_max for item in comparisons])
        )


@dataclasses.dataclass
class EvaluationLapTime:
    truth: float
    predicted: float
    error: float
    abs_error: float
    percentage: float
    error_per_minute: float

    @staticmethod
    def from_values(truth, predicted):
        # TODO Test
        absolute_error = abs(truth - predicted)
        percentage_error = (absolute_error / truth) * 100
        error_per_minute = absolute_error / truth * 60

        return EvaluationLapTime(
            truth=truth,
            predicted=predicted,
            error=truth - predicted,
            abs_error=absolute_error,
            percentage=percentage_error,
            error_per_minute=error_per_minute
        )

    # TODO Test
    @staticmethod
    def combine(comparisons: List['EvaluationLapTime']) -> 'EvaluationLapTime':
        return EvaluationLapTime(
            truth=_mean([item.truth for item in comparisons]),
            predicted=_mean([item.predicted for item in comparisons]),
            error=_mean([item.error for item in comparisons]),
            abs_error=_mean([item.abs_error for item in comparisons]),
            percentage=_mean([item.percentage for item in comparisons]),
            error_per_minute=_mean([item.error_per_minute for item in comparisons])
        )


@dataclasses.dataclass
class Evaluation:
    laptime: EvaluationLapTime

    position: EvaluationError
    velocity: EvaluationError

    apexes: List[int]

    @staticmethod
    def from_errors(
            laptime: float,
            predicted_laptime: float,
            position_deltas: List[float],
            position_percentage_errors: List[float],
            velocity_deltas: List[float],
            velocity_percentage_errors: List[float],
            apexes: List[int]
    ):
        return Evaluation(
            laptime=EvaluationLapTime.from_values(laptime, predicted_laptime),

            position=EvaluationError.from_errors(position_deltas, position_percentage_errors, apexes),
            velocity=EvaluationError.from_errors(velocity_deltas, velocity_percentage_errors, apexes),

            apexes=apexes
        )

    # TODO Test
    @staticmethod
    def combine(comparisons: List['Evaluation']) -> 'Evaluation':
        return Evaluation(
            laptime=EvaluationLapTime.combine([x.laptime for x in comparisons]),
            position=EvaluationError.combine([x.position for x in comparisons]),
            velocity=EvaluationError.combine([x.velocity for x in comparisons]),
            apexes=[]
        )
