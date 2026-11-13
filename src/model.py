import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, width=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, width * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(width * 8), nn.ReLU(True),
            nn.ConvTranspose2d(width * 8, width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 4), nn.ReLU(True),
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(True),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(True),
            nn.ConvTranspose2d(width, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z):
        if z.ndim == 2:
            z = z[:, :, None, None]
        return self.main(z)


class Discriminator(nn.Module):

    def __init__(self, width=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, width, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width, width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width * 4, width * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images):
        return self.main(images).flatten()


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def make_models(latent_dim=100, width=64, device="cpu"):
    generator = Generator(latent_dim, width).to(device)
    discriminator = Discriminator(width).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    return generator, discriminator
