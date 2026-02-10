import json

import numpy as np
from toolkit import maths
from lapsim.eval.evaluation import Evaluation, EvaluationLapTime, EvaluationError
from toolkit.tracks.models import SegmentationLine, Track

from utils.test_base import TestBase
from lapsim import eval


"""Test encoder functionality by testing how inputs affect the output"""


class TestEvaluation(TestBase):

    def get_spliced_data(self):
        """get testing data"""
        with open(self.get_lapsim_data_path() / 'spliced' / '100586536.json') as file:
            data = json.load(file)
            return Track(**data['track'])

    def test_calculate_optimal_positions(self):
        spliced_track = self.get_spliced_data()
        optimal_path = eval.calculate_optimal_positions(spliced_track)
        self.assertTupleEqual((1174, 2), optimal_path.shape)

        # Test that distances on average are near 5m
        distances = [[maths.distance(optimal_path[i], optimal_path[i-1]) for i in range(len(optimal_path))]]
        self.assertAlmostEqual(5.0, float(np.mean(distances)), delta=0.02)

    def test_find_apexes(self):
        spliced_track = self.get_spliced_data()
        apexes = eval.find_apexes(spliced_track.segmentations)

        self.assertListEqual(
            apexes,
            [63, 88, 117, 137, 232, 267, 326, 422, 432, 539, 570, 587, 682, 796, 864, 895, 997, 1103, 1113]
        )

    def test_laptime(self):
        # Create track of two segmentations seperated by 10m (20m round trip)
        # at speeds of 5mps
        laptime = eval.estimate_lap_time(Track(
            segmentations=[
                SegmentationLine(x1=0, y1=0, x2=1, y2=0, pos=0, vel=5),
                SegmentationLine(x1=0, y1=10, x2=1, y2=10, pos=0, vel=5),
            ]
        ))

        # It takes 4 seconds to covert 20m at 5mps
        self.assertEqual(4, laptime)

        # Test on real data
        spliced_track = self.get_spliced_data()
        laptime = eval.estimate_lap_time(spliced_track)

        self.assertAlmostEqual(laptime, 165.8, delta=0.1)

    def test_evaluation(self):
        """Test on a ground truth and predicted from a poorly trained network"""
        truth_data = Track.parse_file(self.get_lapsim_data_path() / 'predicted' / 'ground-0.json')
        pred_data = Track.parse_file(self.get_lapsim_data_path() / 'predicted' / 'predicted-0.json')
        evaluation = eval.evaluate(truth_data, pred_data)

        self.assertAlmostEqual(evaluation.laptime.truth, 89.35, places=1)
        self.assertAlmostEqual(evaluation.laptime.predicted, 101.36, places=1)
        self.assertAlmostEqual(evaluation.laptime.abs_error, 12.0, places=1)
        self.assertAlmostEqual(evaluation.laptime.percentage, 13.4, places=1)  # 13.4% error
        self.assertAlmostEqual(evaluation.laptime.error_per_minute, 8.06, places=1)

        self.assertAlmostEqual(evaluation.position.mean_absolute, 1.88, places=1)  # meteres
        self.assertAlmostEqual(evaluation.position.max, 8.16, places=1)  # meteres
        self.assertAlmostEqual(evaluation.position.percentage_mean, 19.72, places=1)  # %
        self.assertAlmostEqual(evaluation.position.percentage_ci95, 48.62, places=1)  # %

        self.assertAlmostEqual(evaluation.velocity.mean_absolute, 9.19, places=1)  # meteres
        self.assertAlmostEqual(evaluation.velocity.max, 25.12, places=1)  # meteres
        self.assertAlmostEqual(evaluation.velocity.percentage_mean, 12.13, places=1)  # %
        self.assertAlmostEqual(evaluation.velocity.percentage_ci95, 31.15, places=1)  # %

    def test_multi_evaluation(self):
        def load_eval(index: int):
            truth_data = Track.parse_file(self.get_lapsim_data_path() / 'predicted' / f'ground-{index}.json')
            pred_data = Track.parse_file(self.get_lapsim_data_path() / 'predicted' / f'predicted-{index}.json')
            return eval.evaluate(truth_data, pred_data)

        evaluation = Evaluation.combine([load_eval(i) for i in range(10)])

        self.assertAlmostEqual(evaluation.laptime.abs_error, 21.25, places=1)
        self.assertAlmostEqual(evaluation.laptime.percentage, 16.19, places=1)  # 13.4% error
        self.assertAlmostEqual(evaluation.laptime.error_per_minute, 9.71, places=1)

        self.assertAlmostEqual(evaluation.position.mean_absolute, 1.42, places=1)  # meteres
        self.assertAlmostEqual(evaluation.position.max, 8.25, places=1)  # meteres
        self.assertAlmostEqual(evaluation.position.percentage_mean, 13.82, places=1)  # %
        self.assertAlmostEqual(evaluation.position.percentage_ci95, 37.88, places=1)  # %

        self.assertAlmostEqual(evaluation.velocity.mean_absolute, 7.98, places=1)  # meteres
        self.assertAlmostEqual(evaluation.velocity.max, 28.19, places=1)  # meteres
        self.assertAlmostEqual(evaluation.velocity.percentage_mean, 18.83, places=1)  # %
        self.assertAlmostEqual(evaluation.velocity.percentage_ci95, 35.48, places=1)  # %

    def test_null_apex_variables(self):
        """Previously, Nones in apex errors where a track was poorly predicted
        may not ave any apexes, which resulted in None errors. This test checks
        for this to prevent errors"""
        # Test this runs without creating an error
        response = Evaluation.combine(comparisons=[
            Evaluation(
                laptime=EvaluationLapTime(
                    truth=0,
                    predicted=0,
                    error=0,
                    abs_error=0,
                    percentage=0,
                    error_per_minute=0,
                ),
                position=EvaluationError(
                    max=0,
                    mean=0,
                    mean_absolute=0,
                    rmse=0,
                    ci95=0,
                    percentage_mean=0,
                    percentage_max=0,
                    percentage_ci95=0,
                    apex_mean=None,
                    apex_mean_absolute=None,
                    apex_max=None,
                ),
                velocity=EvaluationError(
                    max=0,
                    mean=0,
                    mean_absolute=0,
                    rmse=0,
                    ci95=0,
                    percentage_mean=0,
                    percentage_max=0,
                    percentage_ci95=0,
                    apex_mean=None,
                    apex_mean_absolute=None,
                    apex_max=None,
                ),
                apexes=[1, 2]
            ),
        ])

        self.assertEqual(0, len(response.apexes))
        self.assertIsNone(response.velocity.apex_max)
        self.assertIsNone(response.velocity.apex_mean)
        self.assertIsNone(response.velocity.apex_mean_absolute)
