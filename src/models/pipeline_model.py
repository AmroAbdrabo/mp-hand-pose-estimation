import torch
import torch.nn as nn

from src.models import model_factory


class Pipeline(nn.Module):
    """
    This is a bare bones main model class. This is where you abstract away the exact
    backbone architecture detail and do fancy stuff with the output of the backbone.
    E.g for MANO you may want to construct the MANO mesh based on the parameters
    """

    def __init__(self, pipeline, stages):
        super().__init__()
        # NOTE You can try different backend models here
        
        self.pipeline = pipeline
        self.stages = stages

    def forward(self, batch):
        # Feed through backend model
        input = batch['image']
        for idx, el in enumerate(self.stages):
            mod_el = (self.pipeline[idx])(input)
            input = mod_el[el]

        #output = self.backend_model(batch["image"])
        # Adjust shape of output
        #output["kp3d"] = output["kp3d"].view(-1, 21, 3)
        input = input.view(-1, 21, 3)
        out = {"kp3d": input}
        return out
