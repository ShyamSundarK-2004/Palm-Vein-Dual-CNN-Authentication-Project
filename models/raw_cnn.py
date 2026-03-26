import torch
import torch.nn as nn
from torchvision import models


class RawResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.is_dual = False
        # ResNet18 backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


        # ResNet expects 3-channel input, but we have 1-channel
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace classifier
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
