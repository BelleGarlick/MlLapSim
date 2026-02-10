from lapsim.normalisation.normalised_data import NormalisedData
from lapsim.encoder.partition import Partition
from lapsim.normalisation.transform_normalisation import TransformNormalisation
from lapsim.normalisation.transforms.transformer import Transform
from utils.test_base import TestBase


def create_toy_partition():
    return Partition(
        vehicles=[
            {
                "width": -1,
                "trackFront": 0,
                "trackRear": 1,
                "wheelBaseFront": 2,
                "wheelBaseRear": 3,
                "mass": 4,
                "KDriveFront": 5,
                "KRoll": 6,
                "tyreFriction": 7,
                "maxPower": 8,
                "CoGHeight": 9,
                "FDriveMax": 10,
                "liftCoeffFront": 11,
                "liftCoeffRear": 12,
                "VMax": 13,
                "dragCoeff": 14,
                "yawInertia": 15,
                "KBrakeFront": 16
            }
        ],
        widths=[[15, 16, 17, 18, 19, 20]],
        angles=[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
        offsets=[[0.05, 0.15, 0.25, 0.35, 0.45, 0.55]],
        positions=[[0, 0.2, 0.4, 0.6, 0.8, 1.0]],
        velocities=[[50, 60, 70, 80, 90, 100]]
    )


class TestTransformBase(TestBase):

    @staticmethod
    def load_toy_partition(method: str, **kwargs):
        toy_partition = create_toy_partition()

        bounds = TransformNormalisation(
            transform=Transform(method=method, **kwargs)).extend(toy_partition)

        vehicle_vectors = bounds.transform.vectorise_vehicles(toy_partition.vehicles)
        normalised_data = bounds.bounds.normalise(toy_partition, vehicle_vectors)
        input, output, vehicles = bounds.transform.transform(normalised_data, cores=1)

        return normalised_data, toy_partition, bounds, input, output, vehicles

    def load_real_partition(self, method: str, **kwargs):
        partition = Partition.load(self.get_lapsim_data_path() / 'encoded' / 'partition-1.json')

        bounds = TransformNormalisation(
            transform=Transform(method=method, **kwargs)).extend(partition)

        vehicle_vectors = bounds.transform.vectorise_vehicles(partition.vehicles)
        normalised_data = bounds.bounds.normalise(partition, vehicle_vectors)
        inp, output, vehicles = bounds.transform.transform(normalised_data, cores=1)

        return normalised_data, partition, bounds, inp, output, vehicles

    def assertSamplingCorrect(
            self, normalised_data: NormalisedData, y_pos, y_vel, sampling, lag=0, patch_size: int = 1):
        # TODO Document and clear up
        global_normal_index = 0
        for track_index in range(len(normalised_data.angles)):
            track_length = len(normalised_data.angles[track_index])

            # TODO smartly choose this based on sampling size * patch size & track length
            extended_pos = normalised_data.positions[track_index] * 7
            extended_vel = normalised_data.velocities[track_index] * 7

            for normal_index in range(len(normalised_data.angles[track_index])):
                target_pos_window = y_pos[global_normal_index]
                target_vel_window = y_vel[global_normal_index]

                start = normal_index - lag + 1 - patch_size - (sampling * patch_size)
                start = start % track_length

                end = start + len(target_vel_window)

                self.assertFloatListEqual(target_pos_window, extended_pos[start:end])
                self.assertFloatListEqual(target_vel_window, extended_vel[start:end])

                global_normal_index += 1
