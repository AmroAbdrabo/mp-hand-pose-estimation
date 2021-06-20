import torch

from src.models import torchvision_model_wrapper
from src.models import main_model
from src.models import bou_model


def get_model(cfg_model, dev=None):
    model_name = cfg_model.name.lower()

    if dev is None:
        dev = torch.device("cpu")

    if model_name == "main_model":
        model = main_model.MainModel(cfg_model)
    elif model_name == "bou50":
        model = bou_model.getBou50(ispretrain=False, backend = cfg_model.backend, input_option=0)
    elif model_name == "bou34":
        model = bou_model.getBou34(ispretrain=False, input_option=0) 
    elif model_name in torchvision_model_wrapper.model_list():
        # ResNet
        #print('called')
        model = torchvision_model_wrapper.get_model(cfg_model)
        #print(type(model))
    else:
        # NOTE You may want to test other model types which are not supported by resnet_wrapper
        raise Exception(f"Unsupported model {model_name}")

    # Push to desire device (cpu/gpu)
    if (type(model) is list):
        for el in model:
            el.to(dev)
    else:
        model.to(dev)

    return model

