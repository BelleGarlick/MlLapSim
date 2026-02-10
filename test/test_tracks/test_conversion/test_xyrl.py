from toolkit.tracks.models import Track
from utils.test_base import TestBase

import toolkit as nst


class TestTrackConversionXYRL(TestBase):

    def test_from_xyrl(self):
        """Load up these tracks and check the expected normals is correct"""
        for file_path, expected_normals in [
            ('conversion/xyrl/1.csv', 5799),
            ('conversion/xyrl/2.csv', 3293),
            ('conversion/xyrl/3.csv', 5023)
        ]:
            with open(self.get_tracks_data_path() / file_path) as file:
                track = nst.tracks.conversion.from_xyrl(raw=file.read())
                self.assertEqual(expected_normals, len(track.segmentations))

    def test_to_xyrl(self):
        """Test rebuilding doesn't throw errors. This has also been visually tested"""
        track = Track.from_file(self.get_tracks_data_path() / 'conversion' / 'normals' / '1.json')
        xyrl = nst.tracks.conversion.to_xyrl(track)
        self.assertEqual(len(xyrl), 3577)

        # == Uncomment to visualise, tracks should look the same ==
        # import matplotlib
        # matplotlib.use("TkAgg")
        # import matplotlib.pyplot as plt
        # rebuilt_track = nst.tracks.conversion.from_xyrl(data=xyrl)
        # plt.plot([x[0] for x in track.left_line()], [x[1] for x in track.left_line()])
        # plt.plot([x[0] for x in track.right_line()], [x[1] for x in track.right_line()])
        # plt.plot([x[0] for x in rebuilt_track.left_line()], [x[1] for x in rebuilt_track.left_line()])
        # plt.plot([x[0] for x in rebuilt_track.right_line()], [x[1] for x in rebuilt_track.right_line()])
        # plt.axis("equal")
        # plt.show()
