import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_function = nn.BCELoss()

    def forward(self, x, result):
        return self.loss_function(torch.squeeze(x), result)
