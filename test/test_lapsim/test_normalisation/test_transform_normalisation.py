from lapsim.encoder.partition import Partition
from lapsim.normalisation.transform_normalisation import TransformNormalisation
from utils.test_base import TestBase


"""Test basic features of transform normalisation"""


class TestTransformNormalisationFunctionality(TestBase):

    def test_saving_and_loading(self):
        self.get_temp_output_path().mkdir(parents=True, exist_ok=False)
        output_path = self.get_temp_output_path() / "bounds1.json"

        # Create original bounds
        transform_and_normalisation = TransformNormalisation()
        transform_and_normalisation.transform.vehicle_encoding = [
            "test1", "test2"
        ]
        transform_and_normalisation.transform.method = "test3"
        transform_and_normalisation.transform.lag = 69
        transform_and_normalisation.transform.sampling = 420

        transform_and_normalisation.save(output_path)

        # Test loading and compare
        loaded_bounds = TransformNormalisation.load(output_path)
        self.assertListEqual(
            transform_and_normalisation.transform.vehicle_encoding,
            loaded_bounds.transform.vehicle_encoding)
        self.assertEqual(
            transform_and_normalisation.transform.method,
            loaded_bounds.transform.method)
        self.assertEqual(
            transform_and_normalisation.transform.lag,
            loaded_bounds.transform.lag)
        self.assertEqual(
            transform_and_normalisation.transform.sampling,
            loaded_bounds.transform.sampling)

    def test_basic_normalising(self):
        """Test data is loaded and normalised correctly"""
        path = self.get_lapsim_data_path() / 'encoded' / '100586536.json'
        partition = Partition.load(path)

        transform_and_normalisation = TransformNormalisation()
        transform_and_normalisation.transform.method = "lag"
        transform_and_normalisation.transform.lag = 10
        transform_and_normalisation.transform.sampling = 3
        transform_and_normalisation.extend(partition)

        # Test data was outputed
        x, (y_pos, y_vel), vehicles = transform_and_normalisation.normalise_and_transform(partition)
        self.assertEqual(1174, len(x))
        self.assertEqual(1174, len(vehicles))

        self.assertTupleEqual((2348, 3), x[0].shape)
        self.assertTupleEqual((16, ), vehicles[0].shape)

        self.assertTupleEqual((1174, 7), y_pos.shape)
        self.assertTupleEqual((1174, 7), y_vel.shape)

    def test_async_loader(self):
        """Test data is loaded and normalised correctly after async loading"""
        path = self.get_lapsim_data_path() / 'encoded' / '100586536.json'
        partition = Partition.load(path)

        transform_and_normalisation = TransformNormalisation()
        transform_and_normalisation.transform.method = "window"
        transform_and_normalisation.transform.foresight = 9
        transform_and_normalisation.transform.sampling = 2
        transform_and_normalisation.extend(partition)

        # Test data wasn't loaded
        loader = transform_and_normalisation.async_load_and_normalise_partition(path)
        self.assertIsNone(loader.partition)
        self.assertIsNone(loader.normalisation)

        loader.join()

        # Test partition is loaded
        self.assertIsNotNone(loader.partition)
        self.assertEqual(1, len(loader.partition.widths))
        self.assertEqual(1174, len(loader.partition.widths[0]))

        # Test data was outputed
        x, (y_pos, y_vel), vehicles = loader.normalisation
        self.assertTupleEqual((1174, 3, 19), x.shape)
        self.assertTupleEqual((1174, 16), vehicles.shape)
        self.assertTupleEqual((1174, 5), y_pos.shape)
        self.assertTupleEqual((1174, 5), y_vel.shape)
