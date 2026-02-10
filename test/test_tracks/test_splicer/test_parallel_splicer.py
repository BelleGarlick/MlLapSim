import os
import time
from typing import List

import toolkit.tracks.conversion
from toolkit.utils import readers
from utils.test_base import TestBase
from toolkit.tracks.splicer import SplicerInput, parallel_splice, PathInput


def export_track(track, output_file_path):
    with open(output_file_path, "w+") as file_writer:
        file_writer.write(track.model_dump_json())


class TestParallelConverter(TestBase):

    def get_data_inputs(self, output: bool = False):
        """Get path inputs for testing"""
        data_path = self.get_lapsim_data_path() / 'optimal_tracks'
        output_path = self.get_temp_output_path()

        inputs: List[SplicerInput] = []
        for file in os.listdir(data_path):
            track_path = data_path / file / 'track.csv'
            if not os.path.exists(track_path):
                continue

            with open(track_path) as track_file:
                track = toolkit.tracks.conversion.from_xyrl(track_file.read())

            with open(data_path / file / 'optimal_path.csv') as optim_file:
                optimal_path = readers.read_csv_reader(optim_file.read(), delimiter=";")
                optimal_path = [
                    PathInput(
                        x=optimal_path['x_m'][i],
                        y=optimal_path['y_m'][i],
                        vel=optimal_path['vx_mps'][i],
                        acc=optimal_path['ax_mps2'][i]
                    )
                    for i in range(len(optimal_path['s_m']))
                ]

            inputs.append(SplicerInput(
                track=track,
                path=optimal_path,
                on_complete_args=[output_path / f"{file}.json"],
                on_complete=export_track if output else None
            ))

        return inputs

    def test_parallel_converter(self):
        """Test parallel converter works as expected with diff inputs"""
        # Test works
        inputs = self.get_data_inputs()
        parallel_conversions = parallel_splice(inputs)
        self.assertEqual(9, len(parallel_conversions.spliced))
        self.assertFalse(self.get_temp_output_path().exists())
        self.assertEqual(0, len(parallel_conversions.errors))

        # Test saves correctly and returns
        inputs = self.get_data_inputs(output=True)
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        parallel_conversions = parallel_splice(inputs)
        self.assertEqual(9, len(os.listdir(self.get_temp_output_path())))
        self.assertEqual(9, len(parallel_conversions.spliced))
        self.assertEqual(0, len(parallel_conversions.errors))

        # Test works by saving but not returning
        self.clear_temp_dir()
        inputs = self.get_data_inputs(output=True)
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        parallel_conversions = parallel_splice(inputs, return_output=False)
        self.assertEqual(9, len(os.listdir(self.get_temp_output_path())))
        self.assertEqual(0, len(parallel_conversions.spliced))
        self.assertEqual(0, len(parallel_conversions.errors))

    def test_performance_change(self):
        """Test that multicore processing is faster"""
        inputs = self.get_data_inputs()

        start = time.time()
        parallel_splice(inputs)
        m_core_time = time.time() - start

        start = time.time()
        parallel_splice(inputs, cores=1, batch_size=1)
        s_core_time = time.time() - start

        self.assertLess(m_core_time, s_core_time)
