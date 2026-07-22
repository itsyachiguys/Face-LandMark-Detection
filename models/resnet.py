import torch
import torch.nn as nn
from torchvision.models import resnet34

class FaceLandmarkModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet34(weights="DEFAULT")

        self.backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 136), 
        )

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 68, 2)