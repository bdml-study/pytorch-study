from settings import IMAGE_SIZE
from torch import nn


class ClassifierForCatsAndDogs(nn.Module):

    def __init__(self):
        super(ClassifierForCatsAndDogs, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(IMAGE_SIZE*IMAGE_SIZE*3, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)


class ClassifierForMNIST(nn.Module):

    def __init__(self):
        super(ClassifierForMNIST, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(IMAGE_SIZE*IMAGE_SIZE, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.classifier(x)
