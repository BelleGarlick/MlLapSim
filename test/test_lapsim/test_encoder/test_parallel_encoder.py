import json
import os
import time

from lapsim.encoder.parallel_encoder import create_partition_groups
from lapsim.encoder.partition import Partition
from toolkit.tracks.models import Track
from utils.test_base import TestBase
from lapsim import encoder
from lapsim.encoder import EncoderInput

"""Test features of parallel encoding"""


class TestParallelEncoder(TestBase):

    def get_inputs(self):
        """get testing data"""
        path = self.get_lapsim_data_path() / 'spliced'
        paths = [path / x for x in os.listdir(path) if x[0] != '.']

        inputs = []
        for path in paths:
            with open(path) as file:
                data = json.load(file)
                inputs.append(
                    EncoderInput(
                        track=Track.model_validate(data['track']),
                        vehicle=data['vehicle'],
                    )
                )

        return inputs

    def test_encoder_partitions(self):
        """Test that the correct number of partitions is exported according
        to n_partitions
        """
        inputs = self.get_inputs()

        individual_partitions = encoder.parallel_encode(
            inputs=inputs, n_partitions=0, cores=1, batch_size=1, path=None, return_partitions=True)

        three_partitions = encoder.parallel_encode(
            inputs=inputs, n_partitions=3, cores=1, batch_size=1, path=None, return_partitions=True)

        partitions = encoder.parallel_encode(
            inputs=inputs, n_partitions=1, cores=1, batch_size=1, path=None, return_partitions=True)

        # Test that the partitions are combined as expected
        self.assertEqual(9, len(individual_partitions))
        self.assertEqual(3, len(three_partitions))
        self.assertEqual(1, len(partitions))

        # Check that the sum of individual partitions equals the combined
        self.assertEqual(
            sum([len(x.vehicles) for x in individual_partitions]),
            len(partitions[0].vehicles)
        )

    def test_partitions_exporting(self):
        """Test the different methods of exporting works as expexted"""
        inputs = self.get_inputs()

        # Test returning but not saving
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        partitions = encoder.parallel_encode(
            inputs=inputs, n_partitions=3, cores=1, batch_size=1, path=None, return_partitions=True)
        self.assertEqual(0, len(os.listdir(self.get_temp_output_path())))
        self.assertEqual(3, len(partitions))

        # Test saving and returning
        self.clear_temp_dir()
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        partitions = encoder.parallel_encode(
            inputs=inputs,
            n_partitions=3,
            cores=1,
            batch_size=1,
            path=lambda x: self.get_temp_output_path() / f"p-{x}.json",
            return_partitions=True)
        self.assertEqual(3, len(os.listdir(self.get_temp_output_path())))
        self.assertEqual(3, len(partitions))

        # Test saving but not returning
        self.clear_temp_dir()
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        partitions = encoder.parallel_encode(
            inputs=inputs, n_partitions=3, cores=1, batch_size=1,
            path=lambda x: self.get_temp_output_path() / f"p-{x}.json", return_partitions=False)
        self.assertEqual(3, len(os.listdir(self.get_temp_output_path())))
        self.assertIsNone(partitions)

    def test_encoding_performance(self):
        """Test different performance based on number of cores used"""
        inputs = self.get_inputs()

        # Test returning but not saving
        start = time.time()
        encoder.parallel_encode(
            inputs=inputs, n_partitions=1, cores=2, batch_size=1, path=None, return_partitions=True)
        one_cores_time = time.time() - start

        start = time.time()
        encoder.parallel_encode(
            inputs=inputs, n_partitions=1, cores=4, batch_size=4, path=None, return_partitions=True)
        multi_cores_time = time.time() - start

        self.assertLessEqual(multi_cores_time, one_cores_time)

    def test_balanced_partitions(self):
        """Test that items are balanced accross the partitions"""
        items = create_partition_groups(inputs=[0] * 29, n_partitions=2)
        self.assertEqual(15, len(items[0]))
        self.assertEqual(14, len(items[1]))
        self.assertEqual(2, len(items))

        items = create_partition_groups(inputs=[0] * 29, n_partitions=30)
        self.assertEqual(1, len(items[0]))
        self.assertEqual(1, len(items[-2]))
        self.assertEqual(0, len(items[-1]))
        self.assertEqual(30, len(items))

        items = create_partition_groups(inputs=[0] * 29, n_partitions=5)
        self.assertEqual(6, len(items[-2]))
        self.assertEqual(5, len(items[-1]))
        self.assertEqual(5, len(items))

    def test_async_loader(self):
        """Testloading the partition asyncronously works"""
        loader = Partition.async_load(self.get_lapsim_data_path() / 'encoded' / '100586536.json')
        self.assertIsNone(loader.partition)

        loader.join()
        self.assertIsNotNone(loader.partition)
        self.assertEqual(1, len(loader.partition.widths))
        self.assertEqual(1174, len(loader.partition.widths[0]))
