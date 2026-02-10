import json
from pathlib import Path

import toolkit.tracks.conversion
from toolkit.tracks.models import Track
from toolkit.tracks.splicer.models.splicer_input import PathInput
from toolkit.utils import readers
from toolkit.tracks import splicer
from toolkit.tracks.splicer import SplicerInput
from utils.test_base import TestBase

"""Test splicer functionality by testing how inputs affect the output"""


class TestSplicer(TestBase):

    def get_data_paths(self):
        """get testing data"""
        data_path = self.get_lapsim_data_path() / 'optimal_tracks' / '100586536'

        with open(data_path / 'track.csv') as file:
            track = toolkit.tracks.conversion.from_xyrl(file.read())

        with open(data_path / 'optimal_path.csv') as file:
            optimal_path = readers.read_csv_reader(file.read(), delimiter=";")
            optimal_path = [
                PathInput(
                    x=optimal_path['x_m'][i],
                    y=optimal_path['y_m'][i],
                    vel=optimal_path['vx_mps'][i],
                    acc=optimal_path['ax_mps2'][i]
                )
                for i in range(len(optimal_path['s_m']))
            ]

        return track, optimal_path

    def test_splicer(self):
        """Test using objects as inputs work as expected"""
        track, optimal_path = self.get_data_paths()

        spliced = splicer.splice(
            SplicerInput(
                track=track,
                path=optimal_path
            )
        )

        self.assertIsNotNone(spliced.segmentations[0].vel)
        self.assertEqual(5808, len(spliced.segmentations))

    def test_on_complete(self):
        """Test exporing a track woks as expected via the on_complete callback with args"""
        track, optimal_path= self.get_data_paths()
        output_path = self.get_temp_output_path() / "export.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def on_complete(track: Track, path: Path):
            with open(path, "w+") as file:
                file.write(track.model_dump_json())

        splicer.splice(
            SplicerInput(
                track=track,
                path=optimal_path,
                on_complete_args=[output_path],
                on_complete=on_complete
            )
        )

        with open(output_path) as file:
            data = json.load(file)
            self.assertEqual(5808, len(data['segmentations']))

    def test_precision(self):
        """Test that the output data for low quality is either 0/1 and highq is not"""
        track, optimal_path = self.get_data_paths()

        # calc high-quality spliced data
        highq_spliced = splicer.splice(
            SplicerInput(
                track=track,
                path=optimal_path
            )
        )

        # calc low-quality spliced data
        lowq_spliced = splicer.splice(
            SplicerInput(
                track=track,
                path=optimal_path,
                precision=0
            )
        )

        self.assertNotIn(highq_spliced.segmentations[0].pos, [0, 1])
        self.assertIn(lowq_spliced.segmentations[0].pos, [0, 1])
