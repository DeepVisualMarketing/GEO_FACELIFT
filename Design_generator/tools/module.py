import torch
import random
from torchvision import transforms


def init_zero_noises():
    device = 'cuda'

    noises = [torch.zeros(1, 1, 2 ** 2, 2 ** 2, device=device)]

    for i in range(3, 8 + 1):
        for _ in range(2):
            noises.append(torch.zeros(1, 1, 2 ** i, 2 ** i, device=device))

    return noises


def add_new_tiny_noises(batch_size, noises_single):
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(batch_size, 1, 1, 1).normal_().mul(0.001))
    for noise in noises:
        noise.requires_grad = True
    return noises


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def get_stylegan_transform():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    return transform