import os
import cv2 as cv

import numpy as np

from src.utils.utils import json_load
from src.utils.joints import JointInfo as Joints
from src.dataset.dataset_reader import DatasetReader


class FreiHANDDataset(DatasetReader):
    def __init__(self, split, data_transforms, dataset_path):

        # NOTE You may want to extend functionality such that you train on both
        # validation and training data for the final performance
        dataset_name = "FreiHAND"
        super().__init__(dataset_name, dataset_path, data_transforms)

        k_path = os.path.join(dataset_path, f"{split}_K.json")
        K = np.array(json_load(k_path))
        K = np.tile(K, (4, 1, 1))
        # Convert order to AIT order and convert to milimeters
        if split == "test":
            kp3d = np.zeros((len(K), 21, 3))
        else:
            xyz_path = os.path.join(dataset_path, f"{split}_xyz.json")
            kp3d = self.convert_order(np.array(json_load(xyz_path))) * 1000
            kp3d = np.tile(kp3d, (4, 1, 1))

        self.kp3d = kp3d
        self.K = K
        self.split = split

    def load_sample(self, idx):
        img_path = os.path.join(self.dataset_path, f"{self.split}", "%.8d.jpg" % idx)
        kp3d = self.kp3d[idx]
        K = self.K[idx]
        # Load image
        img = cv.imread(img_path)
        # Convert from BGR to RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        return {"image": img, "kp3d": kp3d, "K": K}

    def __len__(self):
        return len(self.kp3d)

    def convert_order(self, kp3d_from):
        """
        Convert order from FreiHAND to MANO (and then to AIT at the end). Accepts batch and sample input
        """
        # order of Freiburg dataset
        # 0: wrist
        # 1 - 4: thumb[palm to tip], ...
        # 5 - 8: index,
        # 9 - 12: middle
        # 13 - 16: ring
        # 17 - 20: pinky,

        output = np.zeros(shape=kp3d_from.shape, dtype=kp3d_from.dtype)

        output[..., Joints.root_mano, :] = kp3d_from[..., 0, :]
        output[..., Joints.thumb_mcp_mano, :] = kp3d_from[..., 1, :]
        output[..., Joints.thumb_pip_mano, :] = kp3d_from[..., 2, :]
        output[..., Joints.thumb_dip_mano, :] = kp3d_from[..., 3, :]
        output[..., Joints.thumb_tip_mano, :] = kp3d_from[..., 4, :]

        output[..., Joints.index_mcp_mano, :] = kp3d_from[..., 5, :]
        output[..., Joints.index_pip_mano, :] = kp3d_from[..., 6, :]
        output[..., Joints.index_dip_mano, :] = kp3d_from[..., 7, :]
        output[..., Joints.index_tip_mano, :] = kp3d_from[..., 8, :]

        output[..., Joints.middle_mcp_mano, :] = kp3d_from[..., 9, :]
        output[..., Joints.middle_pip_mano, :] = kp3d_from[..., 10, :]
        output[..., Joints.middle_dip_mano, :] = kp3d_from[..., 11, :]
        output[..., Joints.middle_tip_mano, :] = kp3d_from[..., 12, :]

        output[..., Joints.ring_mcp_mano, :] = kp3d_from[..., 13, :]
        output[..., Joints.ring_pip_mano, :] = kp3d_from[..., 14, :]
        output[..., Joints.ring_dip_mano, :] = kp3d_from[..., 15, :]
        output[..., Joints.ring_tip_mano, :] = kp3d_from[..., 16, :]

        output[..., Joints.pinky_mcp_mano, :] = kp3d_from[..., 17, :]
        output[..., Joints.pinky_pip_mano, :] = kp3d_from[..., 18, :]
        output[..., Joints.pinky_dip_mano, :] = kp3d_from[..., 19, :]
        output[..., Joints.pinky_tip_mano, :] = kp3d_from[..., 20, :]

        return output


if __name__ == "__main__":
    """
    Visualize one sample
    """
    import matplotlib.pyplot as plt
    from src.utils.vis_utils import plot_fingers
    from mpl_toolkits.mplot3d import Axes3D
    from src.utils.utils import kp3d_to_kp2d

    dataset_path = "/home/adrian/datasets_tmp/freihand_dataset_MP/"
    split = "test"
    data_transform = None

    freihand_dataset = FreiHANDDataset(split, data_transform, dataset_path)

    idx = 0
    sample = freihand_dataset.load_sample(idx)

    img = sample["image"]
    kp3d = sample["kp3d"]
    K = sample["K"]
    kp2d = kp3d_to_kp2d(kp3d, K)  # Project to 2D using pinhole camera model

    fig = plt.figure(figsize=(12, 5))
    ax_rgb = fig.add_subplot(131)
    ax_3d_1 = fig.add_subplot(132, projection="3d")
    ax_3d_2 = fig.add_subplot(133, projection="3d")
    plot_fingers(kp2d, img_rgb=img, ax=ax_rgb)
    plot_fingers(kp3d, ax=ax_3d_1, view=(-90, -90))  # frontal view
    plot_fingers(kp3d, ax=ax_3d_2)  # top view
    plt.show()
