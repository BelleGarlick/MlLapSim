import random
import shutil
import sys
import os

import matplotlib
import multiprocessing
from toolkit import maths

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from toolkit.maths import segment_intersections
import numpy as np

BATCH_SIZE = 32
sys.path.append(os.path.abspath("src"))

# todo restore the data

data_path = "/Users/belle/Downloads/Version 2/"
# data_path = "/Users/belle/Downloads/evaluate tracks/"
check_dirs = [
    data_path + "training",
    data_path + "validation",
    data_path + "test"
    # data_path + "ArtificialTraining",
    # data_path + "Artificial Validation",
    # data_path + "Artificial Test",
    # data_path + "Real Test"
]

# if an item comes up marked as bad but is good, move it to golden
# error_tracks_path = data_path + "faulty"
error_tracks_path = data_path + "golden"

# /Users/belle/Downloads/evaluate tracks/V2 Output/ArtificialTraining/277794793
# /Users/belle/Downloads/evaluate tracks/V2 Output/ArtificialTraining/463539302
# /Users/belle/Downloads/evaluate tracks/V2 Output/ArtificialTraining/208534064
# /Users/belle/Downloads/evaluate tracks/V2 Output/ArtificialTraining/489956755
# /Users/belle/Downloads/evaluate tracks/V2 Output/ArtificialTraining/138945348

# todo check if it's a corner by comparing the distance between the outer and inner lines.
# if ratio is small enough, then it's a straight. all straights should be halfed
# then do this process again to whittle down the numbers.
#. if problem still occurs then do it to all infected ones



def open_path(f):
    with open(f, "r") as file:
        return file.read()


def open_xyrl_file(f):
    data = open_path(f).split("\n")[1:]

    out_data = []
    for line in data:
        line_data = line
        if "," in line_data: line_data = line_data.split(",")
        if "; " in line_data: line_data = line_data.split("; ")

        out_data.append([float(x) for x in line_data])

    return np.array([x for x in out_data if len(x) > 0][::2])


import toolkit.tracks.conversion


def check_collision(file_data):
    track = toolkit.tracks.conversion.from_xyrl(data=file_data)

    seg_arrs = np.array([seg.arr() for seg in track.segmentations])
    all_intersections = []

    for idx, seg in enumerate(track.segmentations):
        pre_lines = segment_intersections(seg_arrs[idx], seg_arrs[:idx])
        post_lines = segment_intersections(seg_arrs[idx], seg_arrs[idx + 1:])

        if len(pre_lines) > 0 or len(post_lines) > 0:
            all_intersections.append(idx)

    return len(all_intersections) > 0, seg_arrs, file_data, all_intersections


def check_width_change(seg_lines):
    widths = maths.line_lengths(seg_lines)
    width_changes = [max(widths[w] / widths[w-1], widths[w-1] / widths[w]) for w in range(len(widths))]

    rolled_data = np.vstack([
        np.roll(widths, shift=1),
        np.roll(widths, shift=0),
        np.roll(widths, shift=-1)
    ])
    data = np.array([-2, 5, -2]) @ rolled_data

    return np.max(data) > 3.5 * np.max(widths)


def collisions_between_lines(lines, safety_margin=0):
    # todo optimise this based on making sure they're both in segments that overlap
    for i in range(len(lines)):
        pre_lines = lines[i - 10:i - safety_margin]
        post_lines = lines[i+safety_margin+1:i + 10]
        if i <= 1: pre_lines = np.array([])

        if len(maths.segment_intersections(lines[i], pre_lines)) > 0 \
                or len(maths.segment_intersections(lines[i], post_lines)) > 0:
            return True
    return False


def check_boundary_line_collision(seg_lines):
    return collisions_between_lines(maths.points_to_lines(seg_lines[:, :2]), safety_margin=1) \
        or collisions_between_lines(maths.points_to_lines(seg_lines[:, 2:]), safety_margin=1)


def boundary_has_dodgy_angle(boundary_angles, line_lengths, threshold = 1.5):
    def roll_arr(arr):
        return np.vstack([
            np.roll(arr, shift=2),
            np.roll(arr, shift=1),
            np.roll(arr, shift=0),
            np.roll(arr, shift=-1),
            np.roll(arr, shift=-2)
        ]).T

    rolled_boundary = roll_arr(boundary_angles)

    line_lengths = roll_arr(line_lengths)
    line_lengths = (line_lengths - np.expand_dims(np.mean(line_lengths, axis=1), axis=-1)) / np.std(line_lengths)

    kernels = np.array(
        [[0.21499134600162506, -0.31354793906211853, -0.10825781524181366, -0.018643662333488464, -0.202010840177536,
          -0.2528177797794342, -0.03862069919705391, -0.30965080857276917, -0.12093853205442429, -0.04562007263302803],
         [0.18768377602100372, -0.039846934378147125, 0.2980262339115143, 0.07797063887119293, -0.10410331934690475,
          -0.298816978931427, 0.062111370265483856, -0.039694465696811676, -0.30166253447532654, 0.05805963650345802],
         [-0.060399699956178665, 0.030614903196692467, 0.13453605771064758, 0.17561422288417816, -0.2382373958826065,
          0.2665676176548004, 0.10409574955701828, -0.2444770634174347, -0.1873307228088379, -0.07324808835983276],
         [0.25420647859573364, -0.20458944141864777, 0.1574028581380844, -0.04186737537384033, -0.018028561025857925,
          -0.01650005578994751, 0.11837302893400192, 0.006356774363666773, 0.1269170641899109, -0.23430711030960083],
         [-0.03313372656702995, 0.0722731277346611, 0.04156078025698662, 0.06260291486978531, -0.27384164929389954,
          -0.3145909905433655, -0.07080661505460739, -0.28318703174591064, -0.1813499480485916, 0.1190657839179039],
         [0.18544957041740417, 0.27314892411231995, 0.18249286711215973, 0.3023119866847992, -0.1333538293838501,
          -0.309884250164032, 0.23380514979362488, -0.234573632478714, -0.013454573228955269, 0.038958173245191574],
         [0.07832668721675873, 0.06206953153014183, 0.2578285336494446, -0.1583029329776764, -0.2135896235704422,
          0.2747099995613098, -0.29680493474006653, -0.24892622232437134, -0.16488830745220184, 0.12508626282215118],
         [0.29717546701431274, -0.2038368433713913, -0.27997589111328125, -0.10256212204694748, 0.22181129455566406,
          0.29174476861953735, -0.2741810381412506, -0.06847377866506577, -0.053583547472953796, 0.12252478301525116]])
    kernels_b = np.array(
        [0.22866830229759216, 0.10781791061162949, 0.29232412576675415, 0.019853239879012108, 0.1695202738046646,
         0.13524161279201508, -0.03374490141868591, -0.12510007619857788])
    linear = np.array(
        [-0.4822867214679718, -0.3082756996154785, -0.30150556564331055, -0.3433188498020172, -0.1924760639667511,
         -0.409436970949173, -0.12276425957679749, -0.07806485146284103])
    linear_b = np.array(-0.4067060649394989)

    def detect_occurance(arr1, arr2):
        arr = np.hstack([arr1, arr2])
        sig = lambda x: 1 / (1 + np.exp(-x))
        fc1 = sig((kernels @ arr.T).T + kernels_b)
        return sig((linear @ fc1.T) + linear_b)

    likelihood = detect_occurance(rolled_boundary, line_lengths)
    flipped_likelihood = detect_occurance(-rolled_boundary, line_lengths)

    threshold = 0.3

    if np.max(likelihood) > threshold:
        # print()
        worst_idx = np.argmax(likelihood)
        print("\r" + str(np.max(likelihood)))
        print(f"""
  {'{'}
    "angles": {str((rolled_boundary[worst_idx]).tolist())},
    "lengths": {str((line_lengths[worst_idx]).tolist())},
    "y":
  {'}'},
  """)

        return True

    if np.max(flipped_likelihood) > threshold:
        return True

    return False

def process_file(file, plot=True, auto_delete=False):
    boundary_file_path = file + "/" + "boundaries.csv"
    track_file_path = file + "/" + "track.csv"
    optimal_path_path = file + "/" + "optimal_path.csv"

    # optimal_path_file = open(optimal_path_path)

    if not os.path.exists(boundary_file_path) or not os.path.exists(track_file_path):
        print("Missing boundaries.csv or track.csv")
        return True

    boundary_file = open_xyrl_file(boundary_file_path)
    track_file = open_xyrl_file(track_file_path)
    optimal_path = open_xyrl_file(optimal_path_path)

    for f in [boundary_file, track_file]:
        error_reason = False

        seg_lines = toolkit.tracks.conversion.from_xyrl(data=f)
        seg_lines = np.array([x.arr() for x in seg_lines.segmentations])

        left_line = seg_lines[:, :2]
        right_line = seg_lines[:, 2:]

        import maths

        line_lengths = maths.line_lengths(seg_lines)
        angles_left = [maths.angle3(left_line[i-2], left_line[i-1], left_line[i]) for i in range(len(left_line))]
        angles_right = [maths.angle3(right_line[i-2], right_line[i-1], right_line[i]) for i in range(len(right_line))]

        max_angle_change = max(
            np.max(np.abs(angles_left)),
            np.max(np.abs(angles_right)),
        )

        # if not error_reason and check_boundary_line_collision(seg_lines, ):
        #     error_reason = "Boundary line collision"

        if not error_reason and (boundary_has_dodgy_angle(angles_left, line_lengths) or boundary_has_dodgy_angle(angles_right, line_lengths)):
            error_reason = "Boundary angle weight"

        # if not error_reason and max_angle_change > 2.4:
        #     error_reason = f"Max angle change error ({max_angle_change})"

        # if not error_reason and collisions_between_lines(seg_lines):
        #     # boundary_has_dodgy_angle(angles_left, line_lengths)
        #     # boundary_has_dodgy_angle(angles_right, line_lengths)
        #     error_reason = "seg line collision"

        if not error_reason and check_width_change(seg_lines):
            error_reason = "Width change error"

        if error_reason:
            print(f"{error_reason}: {file}")
            for (x1, y1, x2, y2) in seg_lines:
                plt.plot([x1, x2], [y1, y2], linewidth=0.5, color="grey")
            plt.plot(seg_lines[:, 0], seg_lines[:, 1])
            plt.plot(seg_lines[:, 2], seg_lines[:, 3])
            plt.plot(optimal_path[:, 1], optimal_path[:, 2]) # optimal path
            plt.axis("equal")
            plt.show()

            return True
    return False


if __name__ == "__main__":

    # with open("boundaries.json", "r") as f:
    #     boundaries = np.array(json.load(f))
    #
    #     print(boundary_has_dodgy_angle(boundaries))
    #
    # import sys
    # sys.exit(0)

    for dir in check_dirs:
        print(dir)
        simulations = sorted([x for x in os.listdir(dir) if x[0] != "."])
        random.shuffle(simulations)

        with multiprocessing.Pool(processes=16) as pool:

            count = 0
            for i in range(0, len(simulations), BATCH_SIZE):
                print(f"\r{count}/{i}", end="")

                subset = simulations[i:i + BATCH_SIZE]
                args = [dir + "/" + str(x) for x in subset]

                dodge_files = pool.map(process_file, args) \
                    if BATCH_SIZE > 1 else \
                    list(map(process_file, args))

                pairs = zip(subset, dodge_files)
                pairs = [x[0] for x in pairs if x[1]]
                count += len(pairs)

                for sim in pairs:
                    print(sim)
                    input_path = os.path.join(dir, sim)
                    output_path = os.path.join(error_tracks_path, sim)
                    os.makedirs(error_tracks_path, exist_ok=True)
                    shutil.move(input_path, output_path)
