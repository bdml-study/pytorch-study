import torch
from torch import nn
from torchvision.models import resnet18


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        resnet = resnet18(pretrained=True)
        resnet = list(resnet.children())[:8]
        self.features = nn.Sequential(*resnet)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 128, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 1, (4, 4), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        preds = self.classifier(features)
        return preds
