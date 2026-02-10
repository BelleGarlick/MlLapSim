import json
import math

import toolkit
from lapsim.encoder.partition import AsyncPartitionLoader
from toolkit.tracks.models import Track, SegmentationLine
from utils.test_base import TestBase
from lapsim import encoder
from lapsim.encoder import EncoderInput
from lapsim.encoder.encoder import extract_features


"""Test encoder functionality by testing how inputs affect the output"""


class TestEncoder(TestBase):

    def get_data_paths(self):
        """get testing data"""
        return self.get_lapsim_data_path() / 'spliced' / '100586536.json'

    def get_partition_paths(self):
        """get testing data"""
        return self.get_lapsim_data_path() / 'encoded' / 'partition-0.json'

    def get_encoder_input(self, flip=False) -> EncoderInput:
        with open(self.get_data_paths()) as file:
            data = json.load(file)
            return EncoderInput(
                track=Track.model_validate(data["track"]),
                vehicle=data["vehicle"],
                flip=flip
            )

    def test_encoder(self):
        """Test using objects as inputs work as expected"""
        partition = encoder.encode(self.get_encoder_input())
        self.assertEqual(1, len(partition.vehicles))

    def test_extracting_features(self):
        """Test extracting features from a sample 8 blade helicopter set

        This test creates the spiral, then tests that the widths of all
        the lines are the same length, and that the angle is consistent
        around the loop. No offsets applied. This test checks that
        then reverses the order of the lines and checks that the angle
        is the same but negative.
        """
        pos = list(zip(
            [math.cos(i / 4 * math.pi) for i in range(8)],
            [math.sin(i / 4 * math.pi) for i in range(8)]
        ))

        lines = [SegmentationLine(x1=p[0], y1=p[1], x2=0, y2=0) for p in pos]

        widths, angles, offsets = extract_features(lines)
        self.assertTrue(all([round(x, 6) == 1 for x in widths]))
        self.assertTrue(all([round(x + math.tau/8, 7) == 0.0 for x in angles]))
        self.assertTrue(all([round(x, 8) == 0 for x in offsets]))

        # Flip lines and check the angle is the same but flipped
        widths, angles, offsets = extract_features(lines[::-1])
        self.assertTrue(all([round(x, 6) == 1 for x in widths]))
        self.assertTrue(all([round(x - math.tau/8, 7) == 0 for x in angles]))
        self.assertTrue(all([round(x, 8) == 0 for x in offsets]))

    def test_extract_features_on_toy_track(self):
        """ Create another toy track and test that the
        first and fourth line's offset are correct.

        To view run track:
            for line in lines:
                plt.plot([line.x1, line.x2], [line.y1, line.y2])
            plt.show()
        """
        lines = [
            SegmentationLine(x1=1, y1=1, x2=4, y2=-1),
            SegmentationLine(x1=1, y1=2, x2=4, y2=2),
            SegmentationLine(x1=0, y1=3, x2=0, y2=6),
            SegmentationLine(x1=-1, y1=2, x2=-4, y2=2),
            SegmentationLine(x1=-1, y1=2, x2=-4, y2=-2),
            SegmentationLine(x1=-1, y1=-2, x2=-4, y2=-2),
            SegmentationLine(x1=0, y1=-3, x2=0, y2=-6),
            SegmentationLine(x1=1, y1=-2, x2=4, y2=-2)
        ]

        _, _, offsets = extract_features(lines)

        # Test the specified offsets are correct
        self.assertEqual(0, round(offsets[0] - 0.588, 4))
        self.assertEqual(0, round(offsets[4] + 0.9273, 4))

    def test_encoder_flipping(self):
        """Test flipping a track results in the angles and offsets being flipped
        but not the rest the vehicle speed and widths"""
        partition = encoder.encode(self.get_encoder_input())
        flipped_partition = encoder.encode(self.get_encoder_input(flip=True))

        for i in range(len(partition.widths)):
            # Test width and vels are correct
            self.assertEqual(partition.widths[i][0], flipped_partition.widths[i][0])
            self.assertEqual(partition.velocities[i][0], flipped_partition.velocities[i][0])

            # Test position and angles are flipped
            self.assertAlmostEqual(partition.positions[i][0], 1 - flipped_partition.positions[i][0], 6)
            self.assertAlmostEqual(partition.angles[i][0], -flipped_partition.angles[i][0], 6)
            self.assertAlmostEqual(partition.offsets[i][0], -flipped_partition.offsets[i][0], 4)

    def test_encoding_track(self):
        track_path = self.get_lapsim_data_path() / 'optimal_tracks' / '100586536' / 'track.csv'
        with open(track_path) as file:
            track = toolkit.tracks.conversion.from_xyrl(file.read())

        encoded = encoder.encode(
            EncoderInput(
                track=track,
                vehicle={"suspension": 0},
            )
        )

        self.assertListEqual(
            encoded.velocities[0],
            [-1] * 5808
        )
        self.assertListEqual(
            encoded.positions[0],
            [-1] * 5808
        )

    def test_loading_partition(self):
        """Test loading partition asyncrnosly doesn't load it until thread is joined"""
        loader = AsyncPartitionLoader(self.get_partition_paths())
        loader.start()
        self.assertIsNone(loader.partition)
        loader.join()
        self.assertIsNotNone(loader.partition)

        self.assertEqual(675, len(loader.partition.velocities[0]))
        self.assertEqual(1161, len(loader.partition.velocities[1]))
