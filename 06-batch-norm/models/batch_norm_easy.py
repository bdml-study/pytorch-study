import torch
from torch import nn


class BatchNormalization(nn.Module):
    
    def __init__(self, num_features, beta=0.9):
        super(BatchNormalization, self).__init__()
        
        self.beta = beta # for exponential moving average

        # re-shift parameter
        self.shift = nn.parameter.Parameter(
            torch.zeros(1, num_features, 1, 1)
        )
        
        # re-scaling parameter
        self.scale = nn.parameter.Parameter(
            1e-8 + torch.ones(1, num_features, 1, 1)
        )

        self.mean = None
        self.std = None

    def forward(self, x):
        """
        TRAINING PHASE
        1. compute mean & standard deviation of batch
        2. standardization
        3. accumulate mean and standard deviation using moving average
        4. re-scale and re-shift
        
        EVALUATION PHASE
        1. standardization using accumulated mean and variance
        2. re-scale and re-shift
        """
        
        # TRAINIG PHASE
        if self.training:
            # 1. compute mean & standard deviation of batch
            # TODO: replace None with your answer
            mean = None
            std = None
            
            # 2. standardization
            # TODO: replace None with your answer
            x_std = None

            # 3. accumulate mean and standard deviation using moving average
            if self.mean is None:
                self.mean = mean
                self.std = std
            else:
                self.mean = self.mean*self.beta + mean*(1 - self.beta)
                self.std = self.std*self.beta + std*(1 - self.beta)
        # EVALUATION PHASE
        else:
            # 1. standardization using accumulated mean and variance
            # TODO: replace None with your answer
            x_std = None

        # re-scale and re-shift
        # TODO: replace None with your answer
        x_rescaled = None
        return x_rescaled

