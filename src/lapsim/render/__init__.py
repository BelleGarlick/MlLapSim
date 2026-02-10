from typing import List

import numpy as np
from toolkit import maths
from lapsim import eval
from lapsim.render.render_item import RenderItem
from toolkit.tracks.models import SegmentationLine

"""Render plots of the generated track/velocity/laterial deviation to compare
the ground truth against the prediction.

IMPORTANT NOTE:
Matplotlib must not be a dependency of this package. Instead, any package that
wants to use it should install the library before calling these functions. This
is essentially to ensure a small total package size for deploying to lambdas.

As a result, matplotlib must only be imported at a function level, not at the 
top of the file.
"""


def plot_velocities(
        tracks: List[RenderItem],
        distance=5,
        show: bool = True,
        ax=None,
        fig_size=None,
        show_axis: bool = True
):
    """Plot speed traces

    Args:
        tracks: List of render items
        distance: The distance between normals, used to properly label axis
        ax: Axis to render the plot onto
        fig_size: Scale of the plot
        show_axis: If true, the plot legend will be rendered
        show: Can set to false if used to render multiple plots
    """
    import matplotlib.pyplot as plt

    if fig_size is not None:
        plt.figure(figsize=fig_size, dpi=250)

    if ax is None: fig, ax = plt.subplots(1, 1)

    x_indicies = [i * distance for i in range(len(tracks[0].track.segmentations))]

    for data in tracks:
        ax.plot(x_indicies, [x.vel for x in data.track.segmentations], label=data.label, color=data.color)

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (mp2)")

    if show_axis:
        ax.legend()

    if show:
        plt.show()


def plot_lateral_deviation(
        tracks: List[RenderItem],
        distance=5,
        show: bool = True,
        ax=None,
        fig_size=None,
        show_axis: bool = True
):
    """Plot the lateral deviations

    Args:
        tracks: List of render items
        distance: The distance between normals, used to properly label axis
        ax: Axis to render the plot onto
        fig_size: Scale of the plot
        show_axis: If true, the plot legend will be rendered
        show: Can set to false if used to render multiple plots
    """
    import matplotlib.pyplot as plt

    if fig_size is not None:
        plt.figure(figsize=fig_size, dpi=250)

    if ax is None: fig, ax = plt.subplots(1, 1)

    x_indicies = [i * distance for i in range(len(tracks[0].track.segmentations))]
    widths = maths.line_lengths([[line.x1, line.y1, line.x2, line.y2] for line in tracks[0].track.segmentations])

    ax.plot(x_indicies, np.array(widths) / 2, label="Track Boundary", color="black")
    ax.plot(x_indicies, -np.array(widths) / 2, color="black")

    for data in tracks:
        path = [data.track.segmentations[i].pos * widths[i] - widths[i] / 2 for i in range(len(data.track.segmentations))]

        ax.plot(x_indicies, path, label=data.label, color=data.color)

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Deviation (m)")

    if show_axis:
        ax.legend()

    if show:
        plt.show()


def plot_track(
        tracks: List[RenderItem],
        show: bool = True,
        ax=None,
        fig_size=None,
        show_axis: bool = True
):
    """Plot the birds-eye views of the tracks

    Args:
        tracks: List of render items
        ax: Axis to render the plot onto
        fig_size: Scale of the plot
        show_axis: If true, the plot legend will be rendered
        show: Can set to false if used to render multiple plots
    """
    import matplotlib.pyplot as plt

    if fig_size is not None:
        plt.figure(figsize=fig_size, dpi=250)

    if ax is None: fig, ax = plt.subplots(1, 1)

    for i, line in enumerate(tracks[0].track.segmentations):
        ax.plot([line.x1, line.x2], [line.y1, line.y2], color="red" if i == 0 else "grey")

    seg_lines: List[SegmentationLine] = tracks[0].track.segmentations
    ax.plot([x.x1 for x in seg_lines], [x.y1 for x in seg_lines], label="Track Boundary", color="black")
    ax.plot([x.x2 for x in seg_lines], [x.y2 for x in seg_lines], label="Track Boundary", color="black")

    for data in tracks:
        path = eval.calculate_optimal_positions(data.track)

        ax.plot(path[:, 0], path[:, 1], label=data.label, color=data.color)

    # start_pos = np.array((
    #     normals[0][0, 0], normals[0][0, 1]
    # ))
    # second_pos = np.array((
    #     normals[0][1, 0], normals[0][1, 1]
    # ))
    #
    # # Draw triangle to mark direction and start line
    # to_angle, scale = geometry.angle_to(start_pos, second_pos), 3
    # start_points = np.array([[-1, 1], [-1, -1], [4, 0]])
    # tri_pos = normals[0][0, 0:2] + (normals[0][0, 0:2] - normals[0][0, 2:4])
    # start_points = [geometry.rotate(p * scale, to_angle, np.zeros(2)) + tri_pos for p in start_points]
    # plt.gca().add_patch(plt.Polygon(start_points, color="red"))

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Distance (m)")
    ax.axis("equal")

    if show_axis:
        ax.legend()

    if show:
        plt.show()


def plot_full(tracks: List[RenderItem], fig_size=None, title: str = None):
    """Plot the combination of the other plots with subplots

    Args:
        tracks: List of render items
        fig_size: Scale of the plot
        title: Title to plot with
    """
    import matplotlib.pyplot as plt

    if fig_size is not None:
        plt.figure(figsize=fig_size, dpi=250)

    ax0 = plt.subplot2grid((4, 2), (0, 0), rowspan=3, colspan=2)
    ax1 = plt.subplot2grid((4, 2), (3, 0), rowspan=1)
    ax2 = plt.subplot2grid((4, 2), (3, 1), rowspan=1)

    if title:
        plt.title(title)
    plot_track(tracks, show=False, ax=ax0, show_axis=False)
    plot_velocities(tracks, show=False, ax=ax1, show_axis=False)
    plot_lateral_deviation(tracks, show=False, ax=ax2, show_axis=False)

    plt.show()


def decompress(args):
    if len(args) == 2:
        return args + (None, )
    return args
