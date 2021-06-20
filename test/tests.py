import os

from manopth import demo
from manopth.manolayer import ManoLayer

from src.utils.utils import kp3d_to_kp2d, json_load
from src.utils.utils import kp3d_to_kp2d_batch
import torch.nn.functional as f

import torch
import numpy as np

from src.utils.joints import JointInfo as Joints



def test_kp3d_to_kp2d_batch():
    sample = 4
    dim = 10
    a = torch.Tensor(np.random.rand(dim, 21, 3))
    b = torch.Tensor(np.random.rand(dim, 3, 3))

    res1 = kp3d_to_kp2d(a[sample], b[sample])
    res2 = kp3d_to_kp2d_batch(a,b)

    print(res1)
    print(res2[sample])


def test_l1_loss():
    a = torch.Tensor([[1, 1], [2, 2], [3, 3]])
    b = torch.Tensor([[0, 0], [2, 2], [3, 3]])
    print(a)
    print(b)
    # c = f.l1_loss(torch.cdist(a, p=2.0),b)
    c = torch.diag(torch.cdist(a,b,p=2.0))
    print(c)


def convert_to_small_dataset():
    dataset_path = r'S:\Projects\MachinePerception\Data_MP_project1_small'
    # train_K
    k_path = os.path.join(dataset_path, "train_K.json")
    K = np.array(json_load(k_path))
    print(K.shape)
    # train_mano

    # train_verts

    # train_xyz


def convert_order(kp3d_from):
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

def test_manoLayer():
    dataset_path = 'S:/Projects/MachinePerception/Data_MP_project1/'
    split = 'train'
    xyz_path = os.path.join(dataset_path, f"{split}_xyz.json")
    kp3d = convert_order(np.array(json_load(xyz_path))) * 1000
    kp3d = np.tile(kp3d, (4, 1, 1))

    mano_path = os.path.join(dataset_path, f"{split}_mano.json")
    mano_params = np.array(json_load(mano_path))
    mano_params = np.tile(mano_params, (4, 1, 1))

    verts_path = os.path.join(dataset_path, f"{split}_verts.json")
    vertices = np.array(json_load(verts_path))
    vertices = np.tile(vertices, (4, 1, 1))

    idx = 0
    mano_layer = ManoLayer(mano_root=os.environ["MANO"] + '/models/', flat_hand_mean=False, use_pca=False, ncomps=48)
    hand_verts, hand_joints = mano_layer(mano_params[idx, 0, :48], mano_params[idx, 0, 48:58])
    demo.display_hand({'verts': hand_verts, 'joints': hand_joints}, mano_faces=mano_layer.th_faces)


if __name__ == "__main__":
    # test_kp3d_to_kp2d_batch()
    # test_l1_loss()
    # convert_to_small_dataset()
    test_manoLayer()
