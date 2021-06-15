import gzip
import numpy as np
import torch
import os
import json
from tqdm import tqdm
import subprocess
import easydict
import yaml
from src.utils.utils import deconvert_order
from src.utils.utils import pyt2np
from torch.utils.tensorboard import SummaryWriter
from src.models.bou_model import Bottleneck, DeconvBottleneck, BasicBlock

from src.useHeatmaps import useHeatmaps

def get_config(config):
    with open(config, 'r') as stream:
        return easydict.EasyDict(yaml.load(stream))


class Trainer:
    def __init__(
            self,
            model,
            loss_fn,
            optimizer,
            data_loader_train,
            data_loader_val,
            data_loader_test,
            exp_dir,
            dev,
    ):

        self.writer = SummaryWriter(os.environ["MP_EXPERIMENTS"] + '/runs')
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
        self.model = model
        # self.opt = optimizer
        self.params = get_config("config.yml")
        # setup the optimizer
        self.lr = self.params.lr
        self.beta1 = self.params.beta1
        self.beta2 = self.params.beta2
        # p_view = self.model.state_dict()
        self.opt = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                    lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.params.weight_decay)
        self.dev = dev
        self.loss_fn = loss_fn
        self.print_freq = 100  # Update print frequency
        self.save_freq = 1
        self.exp_dir = exp_dir
        # self.bottleneck = Bottleneck(1598, 10).to("cuda")  # should change with batch size
        # self.deconvbn = DeconvBottleneck(40, 21).to("cuda")

    def one_pass(self, data_loader, phase, preds_storage=None):
        """
        Performs one pass on the entire training or validation dataset
        """
        model = self.model
        opt = self.opt
        loss_fn = self.loss_fn

        loss_tot = 0
        losses_all = {}
        for it, batch in enumerate(data_loader):
            # Zero the gradients of any prior passes
            opt.zero_grad()
            # Send batch to device
            self.send_to_device(batch)
            # Forward pass

            # x, x3d = model(batch)
            if useHeatmaps():
                x2d, x3d, param = self.model(batch['heatmaps'])
            else:
                x2d, x3d, param = self.model(batch['image'])

            # joint_2d, mesh_2d = x2d[:, :42], x2d[:, 42:]
            joint_3d, mesh_3d = x3d[:, :21, :], x3d[:, 21:, :]

            # print(joint_3d.shape) #torch.Size([10, 21, 3])
            output = {}
            output['kp3d'] = joint_3d
            output['param'] = param
            # Compute loss
            losses = loss_fn(output, batch, phase)

            if losses["loss"].requires_grad:
                # Backward pass, compute the gradients
                losses["loss"].backward()
                # Update the weights
                opt.step()

            # WARNING: This is only accurate if batch size is constant for all iterations
            # If drop_last=True, then this is not the case for the last batch.
            # loss_tot += losses["loss"].detach()


            for loss_name, loss_value in losses.items():
                if loss_name in losses_all:
                    losses_all[loss_name] += loss_value
                else:
                    losses[loss_name].detach()
                    losses_all[loss_name] = loss_value

            if (it % self.print_freq) == 0:
                self.print_update(losses, it, len(data_loader))
                # NOTE You may want to add visualization and a logger like
                # tensorboard, w&b or comet

            if not preds_storage is None:
                preds_storage.append(output["kp3d"])

        for loss_name, loss_value in losses_all.items():
            losses_all[loss_name] = loss_value / len(data_loader)
        # loss_tot /= len(data_loader)

        return losses_all, preds_storage

    def send_to_device(self, batch):
        for k, v in batch.items():
            batch[k] = v.to(self.dev)

    def print_update(self, losses, it, n_iter):
        str_print = f"Iter: {it:04d}/{n_iter:04d}\t"
        for loss_name, loss_value in losses.items():
            str_print += f"{loss_name}: {loss_value:0.4f}\t"
        str_print = str_print[:-2]  # Remove trailing tab

        print(str_print)

    def train_model(self, n_epochs):
        print('Experiment dir: ' + self.exp_dir)
        for e in range(n_epochs):
            print(f"\nEpoch: {e + 1:04d}/{n_epochs:04d}")
            # Train one epoch
            self.model.train()

            if True:
                print("##### PRE-TRAINING #####")
                phase = "pre-train"
            else:
                print("##### TRAINING #####")
                phase = "train"
            losses, _ = self.one_pass(self.data_loader_train, phase=phase)
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(phase + "/" + loss_name, loss_value, e)
            # Evaluate on validation set
            eval_model = True
            if eval_model:
                with torch.no_grad():
                    self.model.eval()
                    print("##### EVALUATION #####")
                    losses, _ = self.one_pass(self.data_loader_val, phase="eval")
                    for loss_name, loss_value in losses.items():
                        self.writer.add_scalar("Eval/" + loss_name, loss_value, e)

            if (e % self.save_freq) == 0:
                # NOTE You may want to store the best performing model based on the
                # validation set in addition
                torch.save({
                    'optim_state_dict': self.opt.state_dict()
                }, os.path.join(self.exp_dir, f"model_{e:04d}_optim.pt"))
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.exp_dir, f"model_{e:04d}.pt"),
                )

        torch.save(
            self.model.state_dict(), os.path.join(self.exp_dir, f"model_last.pt")
        )
        torch.save({
                    'optim_state_dict': self.opt.state_dict()
        }, os.path.join(self.exp_dir, f"model_last_optim.pt"))

    def test_model(self):
        """
        Runs model on testing data
        """
        print("##### TESTING #####")
        # NOTE If you are saving the best performing model, you may want to first load
        # the its weights before running test
        with torch.no_grad():
            _, preds_storage = self.one_pass(
                self.data_loader_test, phase="test", preds_storage=[]
            )

        
        preds_new = []
        preds = pyt2np(torch.cat(preds_storage, dim=0)).tolist()
        try:
            for el in preds:
                deorder = deconvert_order(np.array(el))
                preds_new.append(deorder.tolist())
        except:
            print("problem with changing the order")

        test_path = os.path.join(self.exp_dir, "test_preds.json")
        print(f"Dumping test predictions in {test_path}")
        with open(test_path, "w") as f:
            json.dump(preds, f)

        # use alternative gzip function to work on windows
        with open(test_path, "rb") as f:
            data = f.read()
            bindata = bytearray(data)
            with gzip.open(test_path + '.gz', 'wb') as f_out:
                f_out.write(bindata)

        test_path = os.path.join(self.exp_dir, "test_preds_correct_order.json")
        print(f"Dumping test predictions in {test_path}")
        with open(test_path, "w") as f:
            json.dump(preds_new, f)

        # use alternative gzip function to work on windows
        with open(test_path, "rb") as f:
            data = f.read()
            bindata = bytearray(data)
            with gzip.open(test_path + '.gz', 'wb') as f_out:
                f_out.write(bindata)
        # subprocess.call(['gzip', test_path])
