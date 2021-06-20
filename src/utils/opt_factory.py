import torch
import torch.optim as optim


def get_optimizer(opt_cfg, parameters):
    # NOTE You may want to add a learning rate scheduler

    opt_name = opt_cfg.name.lower()

    if opt_name == "sgd":
        opt = optim.SGD(parameters, lr=opt_cfg.lr)
    if opt_name == "adam":
        opt = optim.Adam(parameters, lr=opt_cfg.lr)
    else:
        # NOTE You may want to test other optimizers
        raise Exception(f"Unsupported model {opt_name}")

    return opt
