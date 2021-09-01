import torch
from torch import nn
from torchvision.models import resnet18


class FaceMaskDetector(nn.Module):

    def __init__(self):
        super(FaceMaskDetector, self).__init__()

        resnet = resnet18(pretrained=True)
        resnet = list(resnet.children())[:8]

        self.features = nn.Sequential(*resnet)

        self.detector = nn.Sequential(
            nn.Conv2d(512, 1, (2, 2), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        preds = self.detector(features)
        return preds
