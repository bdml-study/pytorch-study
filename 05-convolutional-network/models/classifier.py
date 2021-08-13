from torch import nn


class ClassifierForCatsAndDogs(nn.Module):

    def __init__(self):
        super(ClassifierForCatsAndDogs, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=4, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, (5, 5), stride=4, padding=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, (4, 4), stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.classifier(x)
        x = x.squeeze()
        return x


class ClassifierForMNIST(nn.Module):

    def __init__(self):
        super(ClassifierForMNIST, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 10, (7, 7), stride=1, padding=0),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.classifier(x)
        x = x.squeeze()
        return x
