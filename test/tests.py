import os

from src.utils.utils import kp3d_to_kp2d, json_load
from src.utils.utils import kp3d_to_kp2d_batch
import torch.nn.functional as f

import torch
import numpy as np




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

if __name__ == "__main__":
    # test_kp3d_to_kp2d_batch()
    # test_l1_loss()
    convert_to_small_dataset()
