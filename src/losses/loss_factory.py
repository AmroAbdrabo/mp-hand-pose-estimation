import torch
import torch.nn as nn
import torch.nn.functional as f
from matplotlib import pyplot as plt

from src.utils.utils import procrustes, kp3d_to_kp2d_batch
from src.utils.utils import kp3d_to_kp2d
import numpy as np
import cv2


def compute_param_reg_loss(vec):
    assert vec.shape[1] == 22
    beta_weight = 10 ** 4
    beta = vec[:, -10:]
    theta = vec[:, -16:-10]
    ret = torch.mean(theta ** 2) + beta_weight * torch.mean(beta ** 2)
    return ret


def fancy_unicorn(pred, target, dev):
    kp2d = kp3d_to_kp2d_batch(pred["kp3d"], target["K"]).detach()
    kp2d_gr = kp3d_to_kp2d_batch(target["kp3d"], target["K"]).detach()
    img_size = target["image"].shape[2]
    heatmaps = torch.zeros([img_size, img_size, 24])
    for i in range(0, 21):
        posX = int(max(0, min(kp2d[0,i,0], img_size-1)))
        posY = int(max(0, min(kp2d[0,i,1], img_size-1)))
        heatmaps[posY, posX, i] += 1
    heatmaps_np = heatmaps.cpu().detach().numpy
    heatmaps_np = cv2.GaussianBlur(heatmaps, (11, 11), 0)
    heatmaps_np = heatmaps.sum(axis=2)

    plt.imshow(heatmaps_np)
    plt.show()
    result = f.l1_loss(
        kp2d.to(dev),
        kp2d_gr.to(dev),
    ) / 21

    return result


def get_loss(loss_cfg, dev):
    all_losses = {}
    for loss_name, loss_params in loss_cfg.items():
        loss_name = loss_name.lower()

        if loss_name == "kp3d":
            if loss_params.type == "mse":
                loss = lambda pred, target: f.mse_loss(
                    pred["kp3d"].to(dev), target["kp3d"].to(dev)
                )
            elif loss_params.type == "l1":
                loss = lambda pred, target: f.l1_loss(
                    pred["kp3d"].to(dev), target["kp3d"].to(dev)
                )
            else:
                raise Exception(f"Unknown loss type {loss_params.type}")
        elif loss_name == "2d_joint_loss":
            if loss_params.type == "l1":
                # loss = lambda pred, target: fancy_unicorn(pred, target, dev)
                loss = lambda pred, target: f.l1_loss(
                    kp3d_to_kp2d_batch(pred["kp3d"], target["K"]),
                    kp3d_to_kp2d_batch(target["kp3d"], target["K"])
                ) / 21
        elif loss_name == "reg_loss":
            loss = lambda pred, target: compute_param_reg_loss(pred['param'])
        else:
            raise Exception(f"Unknown loss {loss_name}")

        # Store the weight
        loss.weight = float(loss_params.weight)
        loss.phases = loss_params.phases
        all_losses[loss_name] = loss
    # This is the loss computed by the submission server. Use weight = 0
    # so we do not backpropagate wrt to it. We have other losses for that.
    # (You can, of course, update your model wrt. to this loss. There's nothing
    # wrong with that)
    # WARNING: This is a slow metric to compute due to procrustes. Therefore it is
    # advised to only keep this in the evaluation phase
    all_losses["perf_metric"] = PerfMetric(weight=0.0, phases=["eval"])

    total_loss = TotalLoss(all_losses, dev)

    return total_loss


class PerfMetric(nn.Module):
    """
    This is the loss which is computed on the submission server
    WARNING: This is a very costly metric to compute because of procrustes. It is only
    provided so you know what the exact error metric used by the submission system is.
    You are advised to only use sparingly
    """

    def __init__(self, weight, phases):
        super().__init__()
        self.weight = weight
        self.phases = phases

    def forward(self, pred, target):
        # _, _, _, pred_aligned = procrustes(target["kp3d"], pred["kp3d"])
        # err = ((pred_aligned - target["kp3d"]) ** 2).sum(-1).sqrt().mean()

        kp3d_gt = target["kp3d"] * target["scale"].view(-1, 1, 1)
        # Compute PA-MSE with unscaled ground-truth
        _, _, _, pred_aligned = procrustes(kp3d_gt, pred["kp3d"])
        err = ((pred_aligned - kp3d_gt) ** 2).sum(-1).sqrt().mean()

        # Do not backpropagate wrt to this error metric
        err = err.detach()

        return err


class TotalLoss(nn.Module):
    """
    This module composes all losses as defined in the loss_cfg.
    """

    def __init__(self, all_losses, dev):
        super().__init__()
        self.all_losses = all_losses
        self.dev = dev

    def forward(self, pred, target, phase):

        losses = {}
        tot_loss = torch.tensor(0.0, device=self.dev)
        for loss_name, loss_fn in self.all_losses.items():
            if not phase in loss_fn.phases:
                # We do not want to compute this loss in the current phase
                continue
            c_loss = loss_fn(pred, target)
            # This is the loss we backprop through
            if c_loss.requires_grad and not loss_fn.weight == 0:
                tot_loss += c_loss * loss_fn.weight
            # We store the individual losses unweighted for easier inspection
            losses[loss_name] = c_loss.detach()

        losses["loss"] = tot_loss

        return losses
