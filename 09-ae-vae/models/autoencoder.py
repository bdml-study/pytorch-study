import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            nn.ConvTranspose2d(32, 1, (3, 3), stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.encoder(x)
        x_rec = self.decoder(features)
        return x_rec


class VariationalAutoEncoder(nn.Module):

    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),

            nn.Conv2d(64, 128, (1, 1), stride=1, padding=0)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            nn.ConvTranspose2d(32, 1, (3, 3), stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        stat = self.encoder(x)
        mean = stat[:, :64]
        logvar = stat[:, 64:]

        samples = self._reparameterize(mean, logvar)
        x_rec = self.decoder(samples)

        if self.training:
            loss_kld = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), axis=(1, 2, 3))
            loss_kld = torch.mean(loss_kld)
            return x_rec, loss_kld
        else:
            return x_rec


    def _reparameterize(self, mean, logvar):
        sd = torch.exp(logvar/2)
        z = torch.rand_like(mean).to(mean.device)

        samples = mean + sd*z
        return samples
