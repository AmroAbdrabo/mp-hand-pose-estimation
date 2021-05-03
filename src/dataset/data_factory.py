import torch
from torch.utils.data import ConcatDataset


def get_data_reader(data_cfg, split, data_transforms):
    l_dataset = []
    for data_name, data_param in data_cfg.items():
        if data_name.lower() == "freihand":
            from src.dataset.freihand_mp_dataset import FreiHANDDataset

            l_dataset.append(FreiHANDDataset(split, data_transforms, **data_param))

    if len(l_dataset) == 1:
        return l_dataset[0]
    else:
        # We have multiple datasets
        return ConcatDataset(l_dataset)
