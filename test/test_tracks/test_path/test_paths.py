from toolkit import tracks, maths

from utils.test_base import TestBase


class TestTrackPath(TestBase):

    def get_track(self):
        with open(self.get_lapsim_data_path() / 'optimal_tracks/100586536/track.csv') as file:
            return tracks.conversion.from_xyrl(file.read())

    def test_shortest_path_generation(self):
        """Calculate the shortest paths and minline and check midline is longer"""
        track = self.get_track()
        midline = maths.lerp_points_on_lines([n.arr() for n in track.segmentations], [0.5] * len(track.segmentations))
        shortest_path = tracks.path.shortest_path(track, padding=1)

        # Assert correct number of points and total length is smaller than midline
        self.assertLess(
            sum(maths.line_lengths(maths.points_to_lines(shortest_path.positions))),
            sum(maths.line_lengths(maths.points_to_lines(midline)))
        )
        self.assertEqual(len(shortest_path.positions), 5808)
