import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 🔥 1-channel input
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


# 🔥 SUM FUSION MODEL
class SumFusionModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.is_dual = True  # 🔥 important for training logic

        self.raw_net = ResNet18FeatureExtractor()
        self.enh_net = ResNet18FeatureExtractor()

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, raw, enh):
        fr = self.raw_net(raw)
        fe = self.enh_net(enh)

        fused = fr + fe

        return self.classifier(fused)