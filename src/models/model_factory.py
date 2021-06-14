import torch

from src.models import torchvision_model_wrapper
from src.models import main_model
from src.models import bou_model

from src.useHeatmaps import useHeatmaps

def get_model(cfg_model, dev=None):
    model_name = cfg_model.name.lower()

    if dev is None:
        dev = torch.device("cpu")

    if model_name == "main_model":
        model = main_model.MainModel(cfg_model)
    if model_name == "bou":
        if useHeatmaps():
            model = bou_model.resnet34_Mano(ispretrain=False, input_option=1) # not sure if input_option means with or without hm
        else:
            model = bou_model.resnet34_Mano(ispretrain=False, input_option=0)
    elif model_name in torchvision_model_wrapper.model_list():
        # ResNet
        model = torchvision_model_wrapper.get_model(cfg_model)
    else:
        # NOTE You may want to test other model types which are not supported by resnet_wrapper
        raise Exception(f"Unsupported model {model_name}")

    # Push to desire device (cpu/gpu)
    model.to(dev)

    return model
