import math

import torch
import torch.nn as nn


class WingLoss(nn.Module):
    """
    Wing Loss for Facial Landmark Localization
    Feng et al., CVPR 2018
    """

    def __init__(self, w=10.0, epsilon=2.0):
        super().__init__()

        self.w = w
        self.epsilon = epsilon

        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, prediction, target):

        x = prediction - target

        absolute = torch.abs(x)

        loss = torch.where(
            absolute < self.w,
            self.w * torch.log(1 + absolute / self.epsilon),
            absolute - self.C,
        )

        return loss.mean()