import torch
import torch.nn as nn
from torchvision import models

class RawResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.is_dual = False

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, raw, clahe=None):
        x = raw

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x