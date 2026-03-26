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


# 🔥 ECG SCALAR MODEL (FINAL)
class ECGFusionModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.is_dual = True

        self.raw_net = ResNet18FeatureExtractor()
        self.enh_net = ResNet18FeatureExtractor()

        # 🔥 ECG GATE (SCALAR)
        self.gate = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, raw, enh):

        fr = self.raw_net(raw)
        fe = self.enh_net(enh)

        g = self.gate(fe)         # (B,1)
        fe_gated = fe * g         # adaptive weighting

        fused = fr + fe_gated

        out = self.classifier(fused)

        return out, g