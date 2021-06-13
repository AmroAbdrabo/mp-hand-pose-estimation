import argparse
import os
import datetime

# This lib allows accessing dict keys via `.`. E.g d['item'] == d.item
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from src.models import model_factory
from src.utils import opt_factory
from src.dataset import transforms_factory, data_factory
from src.losses import loss_factory
from src.trainer import Trainer
from src.utils.utils import worker_init, set_seed

main_seed = 0

def worker_init1(x):
    worker_init(x, main_seed)

if __name__ == "__main__":
    #import gc
    #gc.collect()
    #torch.cuda.empty_cache()
    set_seed(main_seed)  # Seed main thread
    num_threads = 4
    dev = torch.device("cuda")
    data_dir = "C:\\Users\\amroa\\MP2021\\Data_MP_project1\\Data_MP_project1"
    #assert(os.path.isdir(data_dir))
    save_dir = "C:\\Users\\amroa\\MP2021\\experiments"
    ######## Set-up experimental directories
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # Create experiment ID
    unique_id = str(datetime.datetime.now().microsecond)
    exp_dir = os.path.join(save_dir, f"exp_{unique_id}")
    # If this fails, there was an ID clash. Hence its preferable to crash than overwrite
    os.mkdir(exp_dir)
    ######## Parse arguments
    # NOTE You may want to add other settings here, such as the various cfg's below.
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="learning rate of optimizer", type=float, default=1e-3)
    parser.add_argument(
        "--n_epochs", help="Number of training epochs", type=int, default=50
    )
    parser.add_argument(
        "--batch_size", help="Batch size for one pass", type=int, default=10
    )
    args = parser.parse_args()
    ######### Set-up model
    n_kp = 21
    model_cfg = edict(
        {
            "name": "bou",
            "backend": {
                "name": "resnet50",  # Defines the backend model type
                "output_slices": {
                    "kp3d": n_kp * 3
                },  # Defines the outputs and their dimensionality
            },
        }
    )
    model = model_factory.get_model(model_cfg, dev)
    ######### Set-up loss function
    # NOTE Make sure you have a loss for each output slice as defined in model_cfg
    loss_cfg = edict(
        {"reg_kp2d_kp3d": {"weight": 1, "type": "mse", "phases": ["train", "eval"]}}
    )
    loss_fn = loss_factory.get_loss(loss_cfg, dev)
    ######### Set-up optimizer
    opt_cfg = edict({"name": "sgd", "lr": args.lr})
    opt = opt_factory.get_optimizer(opt_cfg, model.parameters())
    ######### Set-up data transformation
    transformation_cfg = edict(
        {
            "Resize": {"img_size": (320, 320)},
            "ScaleNormalize": {},  # Scale normalization is allowed
        }
    )
    transformations = transforms_factory.get_transforms(transformation_cfg)
    ######### Set-up data reader and data loader
    data_cfg = edict({"FreiHAND": {"dataset_path": data_dir}})
    data_reader_train = data_factory.get_data_reader(
        data_cfg, split="train", data_transforms=transformations
    )
    data_reader_val = data_factory.get_data_reader(
        data_cfg, split="val", data_transforms=transformations
    )
    data_reader_test = data_factory.get_data_reader(
        data_cfg, split="test", data_transforms=transformations
    )
    data_loader_train = DataLoader(
        data_reader_train,
        batch_size=args.batch_size,
        shuffle=True,  # Re-shuffle data at every epoch
        num_workers=num_threads,  # Number of worker threads batching data
        drop_last=True,  # If last batch not of size batch_size, drop
        pin_memory=False,  # Faster data transfer to GPU
        worker_init_fn=worker_init1,  # Seed all workers. Important for reproducibility
    )
    data_loader_val = DataLoader(
        data_reader_val,
        batch_size=args.batch_size,
        shuffle=False,  # Go through the test data sequentially. Easier to plot same samples to observe them over time
        num_workers=num_threads,  # Number of worker threads batching data
        drop_last=False,  # We want to validate on ALL data
        pin_memory=True,  # Faster data transfer to GPU
        worker_init_fn=worker_init1,
    )
    seq_sampler = SequentialSampler(data_reader_test)
    data_loader_test = DataLoader(
        data_reader_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        sampler=seq_sampler,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=worker_init1,
    )
    ######### Set-up trainer and run training
    trainer = Trainer(
        model,
        loss_fn,
        opt,
        data_loader_train,
        data_loader_val,
        data_loader_test,
        exp_dir,
        dev,
    )
    trainer.train_model(args.n_epochs)
    # Test the model and dump the results to disk
    # NOTE You may want to write a separate script that can evaluate a trained model
    trainer.test_model()
