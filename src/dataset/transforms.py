import torch
import cv2 as cv  # cv is faster than PIL

import numpy as np
from src.utils.joints import JointInfo

# NOTE Try adding data augmentation here


class NumpyToPytorch:
    def __call__(self, sample):
        # Torch take C x H x W whereas np,cv use H x W x C
        img = sample["image"].transpose(2, 0, 1)
        # Convert to float and map to [0,1]
        sample["image"] = img.astype(np.float32) / 255
        # Transfrom from numpy array to pytorch tensor
        for k, v in sample.items():
            sample[k] = torch.from_numpy(v).float()

        return sample


class Resize:
    """
    Resizes the image to img_size
    """

    def __init__(self, img_size):
        self.img_size = tuple(img_size)

    def __call__(self, sample):
        sample["image"] = cv.resize(sample["image"], self.img_size)

        return sample


class ScaleNormalize:
    """
    Scale normalizes the 3D joint position by the MCP bone of the index finger.
    The resulting 3D joint skeleton has an index MCP bone length of 1
    NOTE: This function will throw a warning for the test data, as the ground-truth
    is set to 0. This is because they are not available.
    """

    def __init__(self):
        # NOTE Try taking the bone as parameter and see if scale normalizing other bones
        # affects performance
        pass

    def __call__(self, sample):
        kp3d = sample["kp3d"]
        bone_length = np.linalg.norm(
            kp3d[JointInfo.index_mcp] - kp3d[JointInfo.index_pip]
        )
        kp3d = kp3d / bone_length
        sample["kp3d"] = kp3d
        sample["scale"] = np.array(bone_length)
        return sample
