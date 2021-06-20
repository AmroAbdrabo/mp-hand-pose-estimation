import torch
import torch.nn as nn

from src.models import model_factory


class Pipeline2(nn.Module):
    """
    This is a bare bones main model class. This is where you abstract away the exact
    backbone architecture detail and do fancy stuff with the output of the backbone.
    E.g for MANO you may want to construct the MANO mesh based on the parameters
    """

    def __init__(self, cfg):
        super().__init__()
        # NOTE You can try different backend models here
        self.backend_model =  model_factory.get_model(cfg.backend2).to("cuda")
        # self.second_model =model_factory.get_model(cfg.backend).to("cuda")
        self.third_model = model_factory.get_model(cfg.frontend).to("cuda")

    def forward(self, batch):
        # Feed through backend model
        #print("sizes")
        #print(batch.size())
        output = self.backend_model(batch)
        # output["bot1"] = output["bot1"].view(-1, 3, 64, 128)
        #print(output['bot1'].size())
        # output2 = self.second_model(output['bot1'])
        # output2["bot2"] = output2["bot2"].view(-1, 3, 32, 128)
        output["bot1"] = output["bot1"].view(-1, 3, 64, 128)
        output3 = self.third_model(output['bot1'])
        # Adjust shape of output
        output3["kp3d"] = output3["kp3d"].view(-1, 21, 3)

        return output3
