import math

from lapsim.normalisation.normalisation_bounds import NormalisationBounds
from lapsim.encoder.partition import Partition
from lapsim.normalisation.transform_normalisation import TransformNormalisation
from utils.test_base import TestBase


"""This module tests normalisation works as expected"""


def create_toy_partition():
    return Partition(
        vehicles=[],
        widths=[[2.3, 2.5], [2.4, 2.6]],
        angles=[[0, 1], [-2, -3]],
        offsets=[[0, 0.1], [-0.2, -0.3]],
        positions=[[], []],
        velocities=[[0.5, 0.4], [0.7]]
    )


def create_toy_partitions():
    return [
        [0.1, 0, -2, 23, -0.01],
        [0.5, -0.2, 0.2, 0.4, 0.03],
        [1.1, 3, -1, -2.3, 0],
    ]


class TestNormalisationBounds(TestBase):

    def test_extending_empty_bounds(self):
        """This tests the bounds were not affected by extending by an
        empty partition"""
        bounds = NormalisationBounds()

        # Check values are at the default
        self.assertListEqual(
            [bounds.max_angle, bounds.max_offset,
             bounds.max_velocity, bounds.min_velocity,
             bounds.max_width, bounds.min_width,
             bounds.max_vehicle, bounds.min_vehicle],
            [0, 0, -math.inf, math.inf, -math.inf, math.inf, None, None]
        )

        # Test empty partition doesn't affect the values
        bounds.extend(
            Partition(
                vehicles=[{}],
                widths=[[]], angles=[[], []], offsets=[[], [], []],
                positions=[[], []], velocities=[[], [], []]
            ),
            [[], [], []]
        )

        # Check no bugs were caused and nothing was affected
        self.assertListEqual(
            [bounds.max_angle, bounds.max_offset,
             bounds.max_velocity, bounds.min_velocity,
             bounds.max_width, bounds.min_width,
             bounds.max_vehicle, bounds.min_vehicle],
            [0, 0, -math.inf, math.inf, -math.inf, math.inf, None, None]
        )

    def test_extending_bounds(self):
        """Test extending normalisation"""
        bounds = NormalisationBounds()

        # Test empty partition doesn't affect the values
        bounds.extend(create_toy_partition(), create_toy_partitions())

        self.assertEqual(bounds.max_angle, 3)
        self.assertEqual(bounds.max_offset, 0.3)
        self.assertEqual(bounds.min_velocity, 0.4)
        self.assertEqual(bounds.max_velocity, 0.7)
        self.assertEqual(bounds.min_width, 2.3)
        self.assertEqual(bounds.max_width, 2.6)

        self.assertListEqual(bounds.max_vehicle, [1.1, 3.0, 0.2, 23, 0.03])
        self.assertListEqual(bounds.min_vehicle, [0.1, -0.2, -2.0, -2.3, -0.01])

    def test_toy_with_no_extension(self):
        """Test toy normalisation without being extended"""
        bounds = NormalisationBounds()

        toy_partition, toy_vehicles = create_toy_partition(), create_toy_partitions()

        # An exception should be raised since the bounds aren't extended yet
        with self.assertRaises(Exception):
            bounds.normalise(toy_partition, toy_vehicles)

    def test_toy_normalisation(self):
        """Test toy normalisation works as expected"""
        bounds = NormalisationBounds()

        toy_partition, toy_vehicles = create_toy_partition(), create_toy_partitions()

        # Test empty partition doesn't affect the values
        bounds.extend(toy_partition, toy_vehicles)

        normalised_data = bounds.normalise(toy_partition, toy_vehicles)

        # Test the bounds are normalised
        self.assertEqual(1, max([max([abs(x) for x in y]) for y in normalised_data.angles]))
        self.assertEqual(1, max([max([abs(x) for x in y]) for y in normalised_data.offsets]))

        self.assertEqual(0, min([min(y) for y in normalised_data.widths]))
        self.assertEqual(1, max([max(y) for y in normalised_data.widths]))

        self.assertEqual(0, min([min(y) for y in normalised_data.velocities]))
        self.assertEqual(1, max([max(y) for y in normalised_data.velocities]))

        self.assertEqual(0, min([min(y) for y in normalised_data.vehicles]))
        self.assertEqual(1, max([max(y) for y in normalised_data.vehicles]))

    def test_normalisation_on_real_partitions(self):
        """Test on real partitions"""
        # Load two testing partitions and check they're the right size
        partition_1 = Partition.load(self.get_lapsim_data_path() / 'encoded' / 'partition-0.json')
        partition_2 = Partition.load(self.get_lapsim_data_path() / 'encoded' / 'partition-1.json')
        self.assertEqual([2, 3], [len(partition_1.angles), len(partition_2.angles)])

        # Test extending the bounds by different number of track and sizes
        # Note: using the TransformNormalisation since it always vectorises the vehicle
        bounds = TransformNormalisation()
        bounds.extend(partition_1)
        bounds.extend(partition_2)

        vp1 = bounds.transform.vectorise_vehicles(partition_1.vehicles)
        norm_p1 = bounds.bounds.normalise(partition_1, vp1)

        vp2 = bounds.transform.vectorise_vehicles(partition_2.vehicles)
        norm_p2 = bounds.bounds.normalise(partition_2, vp2)

        for norm_p in [norm_p1, norm_p2]:
            for track_idx in range(len(norm_p.angles)):
                self.assertLessEqual(-1, min(norm_p.angles[track_idx]))
                self.assertGreaterEqual(1, max(norm_p.angles[track_idx]))
                self.assertLessEqual(-1, min(norm_p.offsets[track_idx]))
                self.assertGreaterEqual(1, max(norm_p.offsets[track_idx]))

                self.assertLessEqual(0, min(norm_p.widths[track_idx]))
                self.assertGreaterEqual(1, max(norm_p.widths[track_idx]))

                self.assertLessEqual(0, min(norm_p.positions[track_idx]))
                self.assertGreaterEqual(1, max(norm_p.offsets[track_idx]))

                self.assertLessEqual(0, min(norm_p.velocities[track_idx]))
                self.assertGreaterEqual(1, max(norm_p.velocities[track_idx]))

                self.assertLessEqual(0, min(norm_p.vehicles[track_idx]))
                self.assertGreaterEqual(1, max(norm_p.vehicles[track_idx]))
