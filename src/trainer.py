import torch
import os
import json
from tqdm import tqdm
import subprocess

from src.utils.utils import pyt2np


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

        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
        self.model = model
        self.opt = optimizer
        self.dev = dev
        self.loss_fn = loss_fn
        self.print_freq = 100  # Update print frequency
        self.save_freq = 10
        self.exp_dir = exp_dir

    def one_pass(self, data_loader, phase, preds_storage=None):
        """
        Performs one pass on the entire training or validation dataset
        """
        model = self.model
        opt = self.opt
        loss_fn = self.loss_fn

        loss_tot = 0
        for it, batch in enumerate(data_loader):
            # Zero the gradients of any prior passes
            opt.zero_grad()
            # Send batch to device
            self.send_to_device(batch)
            # Forward pass
            output = model(batch)
            # Compute loss
            losses = loss_fn(output, batch, phase)

            if losses["loss"].requires_grad:
                # Backward pass, compute the gradients
                losses["loss"].backward()
                # Update the weights
                opt.step()

            # WARNING: This is only accurate if batch size is constant for all iterations
            # If drop_last=True, then this is not the case for the last batch.
            loss_tot += losses["loss"].detach()

            if (it % self.print_freq) == 0:
                self.print_update(losses, it, len(data_loader))
                # NOTE You may want to add visualization and a logger like
                # tensorboard, w&b or comet

            if not preds_storage is None:
                preds_storage.append(output["kp3d"])

        loss_tot /= len(data_loader)

        return loss_tot, preds_storage

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

        for e in range(n_epochs):
            print(f"\nEpoch: {e+1:04d}/{n_epochs:04d}")
            # Train one epoch
            self.model.train()
            print("##### TRAINING #####")
            self.one_pass(self.data_loader_train, phase="train")
            # Evaluate on validation set
            with torch.no_grad():
                self.model.eval()
                print("##### EVALUATION #####")
                self.one_pass(self.data_loader_val, phase="eval")

            if (e % self.save_freq) == 0:
                # NOTE You may want to store the best performing model based on the
                # validation set in addition
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.exp_dir, f"model_{e:04d}.pt"),
                )

        torch.save(
            self.model.state_dict(), os.path.join(self.exp_dir, f"model_last.pt")
        )

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

        preds = pyt2np(torch.cat(preds_storage, dim=0)).tolist()

        test_path = os.path.join(self.exp_dir, "test_preds.json")
        print(f"Dumping test predictions in {test_path}")
        with open(test_path, "w") as f:
            json.dump(preds, f)
        subprocess.call(['gzip', test_path])


