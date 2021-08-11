import torch
from torch import nn
from models.batch_norm import BatchNormalization


class ClassifierWithoutBatchNorm(nn.Module):

    def __init__(self):
        super(ClassifierWithoutBatchNorm, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=4, padding=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, (5, 5), stride=4, padding=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, (4, 4), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        N = x.size(0)
        preds = self.classifier(x)
        preds = preds.view(N, -1)
        return preds


class ClassifierWithBatchNorm(nn.Module):

    def __init__(self):
        super(ClassifierWithBatchNorm, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, (5, 5), stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, (4, 4), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        N = x.size(0)
        preds = self.classifier(x)
        preds = preds.view(N, -1)
        return preds


class ClassifierWithCustomBatchNorm(nn.Module):

    def __init__(self):
        super(ClassifierWithCustomBatchNorm, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=4, padding=2),
            BatchNormalization(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, (5, 5), stride=4, padding=2),
            BatchNormalization(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, (4, 4), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        N = x.size(0)
        preds = self.classifier(x)
        preds = preds.view(N, -1)
        return preds
