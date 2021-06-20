from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def display_pred_target_hand(hand_info_pred, hand_info_target, batch_index, scale):
    fig = plt.figure()
    ax11 = fig.add_subplot(231, projection='3d')
    ax21 = fig.add_subplot(234, projection='3d')
    ax12 = fig.add_subplot(232, projection='3d')
    ax22 = fig.add_subplot(235, projection='3d')
    ax13 = fig.add_subplot(233, projection='3d')
    ax23 = fig.add_subplot(236, projection='3d')
    display_hand(hand_info_pred, batch_index, ax11)
    ax11.title.set_text('Prediction')
    display_hand(hand_info_target, batch_index, ax21)
    ax21.title.set_text('ground truth, ' + str(int(scale[batch_index].item())))

    display_hand(hand_info_pred, batch_index + 1, ax12)
    ax12.title.set_text('Prediction')
    display_hand(hand_info_target, batch_index + 1, ax22)
    ax22.title.set_text('ground truth, ' + str(int(scale[batch_index].item())))

    display_hand(hand_info_pred, batch_index + 2, ax13)
    ax13.title.set_text('Prediction')
    display_hand(hand_info_target, batch_index + 2, ax23)
    ax23.title.set_text('ground truth, ' + str(int(scale[batch_index].item())))
    plt.show()


def display_hand(hand_info, batch_idx, ax, mano_faces=None):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
        batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=1.0)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    cam_equal_aspect_3d(ax, verts.numpy())
    return ax


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)