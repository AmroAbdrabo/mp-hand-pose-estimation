import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_fingers(kp, **kwargs):
    assert len(kp.shape) == 2

    if kp.shape[1] == 2:
        # 2D keypoints
        plot_fingers2D(kp, **kwargs)
    elif kp.shape[1] == 3:
        # 3D keypoints
        plot_fingers3D(kp, **kwargs)
    else:
        raise Exception(f"Invalid keypoint dimensionality: {kp.shape[1]}")


def plot_fingers2D(kp2d, img_rgb=None, ax=None, c="gt"):
    """
    Plots the 2D keypoints over the image. 
    """

    assert len(kp2d.shape) == 2, "plot_fingers2D does not accept batch input"

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    if not img_rgb is None:
        ax.clear()
        ax.imshow(img_rgb)

    if c == "pred":
        c = ["#660000", "#b30000", "#ff0000", "#ff4d4d", "#ff9999"]
    elif c == "gt":
        c = ["#000066", "#0000b3", "#0000ff", "#4d4dff", "#9999ff"]
    else:
        assert isinstance(c, list)

    for i in range(5):
        idx_to_plot = np.arange(i + 1, 21, 5)
        to_plot = np.concatenate((kp2d[0:1], kp2d[idx_to_plot]), axis=0)
        ax.plot(to_plot[:, 0], to_plot[:, 1], "x-", color=c[i])

    return ax


def plot_fingers3D(kp3d, ax=None, c="gt", lims=None, clear=True, view=(-90, 0)):
    """
    Plots the 3D keypoints over the image.
    """

    assert len(kp3d.shape) == 2, "plot_fingers does not accept batch input"

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if c == "pred":
        c = ["#660000", "#b30000", "#ff0000", "#ff4d4d", "#ff9999"]
    elif c == "gt":
        c = ["#000066", "#0000b3", "#0000ff", "#4d4dff", "#9999ff"]
    else:
        assert isinstance(c, list)

    min_range = -3
    max_range = 3
    for i in range(5):
        idx_to_plot = np.arange(i + 1, 21, 5)
        to_plot = np.concatenate((kp3d[0:1], kp3d[idx_to_plot]), axis=0)
        ax.plot(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], "x-", color=c[i])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if not lims is None:
        min_range, max_range = lims
        ax.set_xlim(min_range, max_range)
        ax.set_ylim(min_range, max_range)
        ax.set_zlim(min_range, max_range)

    if not view is None:
        """
        view=(-90,0): View from above to see depth error more clearly
        view=(-90,-90): View from front to see camera view. 
        It is very similar to plotting 2D keypoints, hence less informative if plotting
        2D keypoints already.
        """
        azim, elev = view
        ax.view_init(azim=azim, elev=elev)

    set_axes_equal(ax)

    return ax


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    Source: https://stackoverflow.com/a/31364297/1550099
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
