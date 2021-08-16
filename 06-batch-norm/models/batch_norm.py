import torch
from torch import nn


class BatchNormalization(nn.Module):
    
    def __init__(self, num_features, beta=0.9):
        super(BatchNormalization, self).__init__()
        
        self.beta = beta

        self.shift = nn.parameter.Parameter(
            torch.zeros(1, num_features, 1, 1)
        )
        self.scale = nn.parameter.Parameter(
            1e-8 + torch.ones(1, num_features, 1, 1)
        )

        self.mean = None
        self.std = None

    def forward(self, x):
        if self.training:
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            # std = torch.std(x, dim=(0, 2, 3), keepdim=True)
            var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)
            std = torch.sqrt(var + 1e-8)
            x_std = (x - mean) / (std + 1e-8)

            if self.mean is None:
                self.mean = mean
                self.std = std
            else:
                self.mean = self.mean*self.beta + mean*(1 - self.beta)
                self.std = self.std*self.beta + std*(1 - self.beta)
        else:
            x_std = (x - self.mean) / (self.std + 1e-8)

        x_rescaled = self.scale * x_std + self.shift
        return x_rescaled

