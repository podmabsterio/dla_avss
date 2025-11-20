import torch
from torch import nn


class LossMix(nn.Module):
    def __init__(self, loss1, loss1_name, loss2, loss2_name, loss2_weight=1.0):
        super().__init__()
        self.loss1 = loss1
        self.loss1_name = loss1_name
        self.loss2 = loss2
        self.loss2_name = loss2_name
        self.loss2_weight = loss2_weight

    def forward(self, **batch):
        loss1_value = self.loss1(**batch)["loss"]
        loss2_value = self.loss2(**batch)["loss"]
        loss = loss1_value + self.loss2_weight * loss2_value
        return {
            "loss": loss,
            self.loss1_name: loss1_value,
            self.loss2_name: loss2_value,
        }
