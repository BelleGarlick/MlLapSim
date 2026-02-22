import os

from toolkit.tracks.conversion import from_xyrl
from toolkit.tracks.smoother.smoother import _extend_normals_until_collision, _collapse_collisions_pairs

from toolkit.tracks.smoother import _split_normals


path = '/Users/belle/Developer/Hyperdrive-Labs-LapSim/datasets/Final Release/Real Tracks XYRL'


def cut_normals(normals, left_xy, right_xy, do_thing=False):
    # Extend normals and find all collisions they half normals make with the boundary
    left_normals, right_normals = _split_normals(normals)

    left_normal_collisions = _extend_normals_until_collision(left_normals, left_xy)
    right_normal_collisions = _extend_normals_until_collision(right_normals, right_xy)

    # Collapse all collisions into one collision per normal
    left_collisions = _collapse_collisions_pairs(left_normals, left_normal_collisions, len(left_xy))
    right_collisions = _collapse_collisions_pairs(right_normals, right_normal_collisions, len(right_xy))

    return [
        left_collisions[i] + right_collisions[i]
        for i in range(len(left_collisions))
    ]


if __name__ == "__main__":
    project_files = sorted(os.listdir(path))

    for i, file_name in enumerate(project_files):
        print(i, file_name)

        if not file_name.endswith(".csv"):
            continue

        import csv

        data = []
        with open(os.path.join(path, file_name), newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in list(spamreader)[1:]:
                data.append([float(x) for x in row])

        track = from_xyrl(data=data)

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.use('macosx')

        plt.plot([x[0] for x in data], [x[1] for x in data])
        plt.plot([x[0] for x in track.left_line()], [x[1] for x in track.left_line()])
        plt.plot([x[0] for x in track.right_line()], [x[1] for x in track.right_line()])
        for n in track.segmentations:
            plt.plot(n.arr()[0::2], n.arr()[1::2])
        plt.axis("equal")
        plt.show()
