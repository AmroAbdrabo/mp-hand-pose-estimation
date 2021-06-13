import torch
import cv2 as cv  # cv is faster than PIL

import numpy as np
from src.utils.joints import JointInfo

import os

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

# Rotate randomly
class Rotate:
    """
    Create Random rotation between -180 and 180
    """

    def __init__(self):
        pass

    def __call__(self, sample):

        kp3d = sample["kp3d"]

        rand_angle = np.random.randint(-180, 180)
        #print(rand_angle)
        angle = rand_angle
        row, col, depth = sample["image"].shape
        center = tuple(np.array([row, col]) / 2)
        rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
        radians = np.deg2rad(angle)
        rot_mat_3D = np.array([[np.cos(radians), -np.sin(radians), 0],
                               [np.sin(radians), np.cos(radians), 0],
                               [0, 0, 1]])

        sample["image"] = cv.warpAffine(sample["image"], rot_mat, (col, row))

        sample["kp3d"] = np.matmul(kp3d, rot_mat_3D)


        return sample


class Flip:
    """
    Flip image with 50/50 chance
    """

    def __init__(self):
        pass

    def __call__(self, sample):

        flipBool = np.random.choice([True, False])
        if(flipBool):
            image = sample["image"]
            kp3d = sample["kp3d"]

            image_flipped = cv.flip(image, 1)
            kp3d_flipped = kp3d
            kp3d_flipped[:, 0] = -kp3d[:, 0]

            sample["image"] = image_flipped
            sample["kp3d"] = kp3d_flipped

        return sample

class AddClutter:
    """
    Add random shapes to simulate occlusion
    """

    def __init__(self):
        self.counter = 0


    def __call__(self, sample):

        image = sample["image"]
        image_size = image.shape[0]

        shape_minsize = 10
        border = image_size / 6
        posX = np.random.randint(border, image_size - border)
        posY = np.random.randint(border, image_size - border)

        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)

        shape_int = np.random.randint(3)

        if (shape_int == 0):
            # Circle
            radius = np.random.randint(shape_minsize, image_size / 6)
            cv.circle(image, (posX, posY), radius, (r, g, b), thickness=-1)
        elif (shape_int == 1):
            # Ellipse
            radius1 = np.random.randint(shape_minsize, image_size / 6)
            radius2 = np.random.randint(shape_minsize, image_size / 6)
            angle = np.random.randint(360)
            cv.ellipse(image, (posX, posY), (radius1, radius2), angle, 0, 360, (r, g, b), thickness=-1)
        elif (shape_int == 2):
            # Rectangle
            len1 = np.random.randint(shape_minsize, image_size / 8)
            len2 = np.random.randint(shape_minsize, image_size / 8)
            cv.rectangle(image, (posX - len1, posY - len2), (posX + len1, posY + len2), (r, g, b), thickness=-1)

        sample["image"] = image

        self.counter = self.counter + 1

        return sample


class ChangeBackground:
    """
    Replaces the background
    """

    def __init__(self):
        self.background_folder = os.environ["MP_BACKGROUNDS"]
        self.green_thresh_low = np.array([0, 120, 0])
        self.green_thresh_high = np.array([255, 255, 255])
        self.counter = 0

        pass

    def __call__(self, sample):

        augment = np.random.choice([True,False])

        if augment:

            image = sample["image"]

            image_size = image.shape[0]

            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

            green = hsv[:, :, 1]

            hsv = cv.GaussianBlur(hsv, ksize=(5, 5), sigmaX=1)
            mask = cv.inRange(hsv, self.green_thresh_low, self.green_thresh_high)

            if (np.sum(np.array(green) >= 128) > 20000):
                # Load random background image
                inner_folder = np.random.choice(os.listdir(self.background_folder))
                while (inner_folder == ".DS_Store"):
                    inner_folder = np.random.choice(os.listdir(self.background_folder))
                bg_file = np.random.choice(os.listdir(self.background_folder + inner_folder))
                bg = cv.imread(self.background_folder + inner_folder + "/" + bg_file)  # [0:image_size, 0:image_size]
                bg = cv.resize(bg, (image_size, image_size))
                bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)

                # compute masks
                mask_inverted = cv.bitwise_not(mask)
                masked_hand = cv.bitwise_and(image, image, mask=mask_inverted)
                masked_bg = cv.bitwise_and(bg, bg, mask=mask)
                hand_bg = cv.bitwise_or(masked_hand, masked_bg)
                hand_bg_rgb = cv.cvtColor(hand_bg, cv.COLOR_BGR2RGB)

                sample["image"] = hand_bg
        return sample
