import json
import random
import numpy as np
import torch


def json_load(p):
    with open(p, "r") as fi:
        d = json.load(fi)
    return d


def kp3d_to_kp2d(kp3d, K):
    """
    Pinhole camera model projection
    K: camera intrinsics (3 x 3)
    kp3d: 3D coordinates wrt to camera (n_kp x 3)
    """
    kp2d = (kp3d @ K.T) / kp3d[..., 2:3]

    return kp2d[..., :2]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id, main_seed):
    seed = worker_id + main_seed
    set_seed(seed)


def pyt2np(tensor):
    return tensor.detach().cpu().numpy()


def procrustes(X, Y):
    """
    A batch-wise pytorch implementation of the PMSE metric.
    Computes the affine transformation from Y to X via procrustes.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Arguments

    X: torch.tensor of shape BS x N x M
    Y: torch.tensor of shape BS x N x M

    where BS: Batch size, N: number of points and M: dim of points

    Returns:
        Rt: Transposed rotation matrix
        s: Scaling factor
        t: Translation factor
        Z: s*matmul(Y,Rt) + t
    """

    if torch.all(X == 0):
        print("X contains only NaNs. Not computing PMSE.")
        return np.nan, Y
    if torch.all(Y == 0):
        print("Y contains only NaNs. Not computing PMSE.")
        return np.nan, Y

    muX = X.mean(dim=1, keepdim=True)
    muY = Y.mean(dim=1, keepdim=True)
    # Center to mean
    X0 = X - muX
    Y0 = Y - muY
    # Compute frobenius norm
    ssX = (X0 ** 2).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
    ssY = (Y0 ** 2).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
    normX = torch.sqrt(ssX)
    normY = torch.sqrt(ssY)
    # Scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY
    # Compute optimum rotation matrix of Y
    A = torch.matmul(X0.transpose(2, 1), Y0)
    U, s, V = torch.svd(A)
    T = torch.matmul(V, U.transpose(2, 1))
    # Make sure we have a rotation
    detT = torch.det(T)
    V[:, :, -1] *= torch.sign(detT).view(-1, 1)
    s[:, -1] *= torch.sign(detT)
    T = torch.matmul(V, U.transpose(2, 1))

    traceTA = s.sum(dim=1).view(-1, 1, 1)

    b = traceTA * normX / normY
    Z = normX * traceTA * torch.matmul(Y0, T) + muX

    c = muX - b * torch.matmul(muY, T)

    Rt = T.detach()
    s = b.detach()
    t = c.detach()

    return Rt, s, t, Z
