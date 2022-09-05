import torch
import math
import numpy as np


def get_avged_W(t_batch_len, latent_mean, DEVICE='cuda'):
    return latent_mean.detach().clone().repeat(t_batch_len, 1).unsqueeze(1).to(DEVICE)


def get_random_latent(g_ema, args, batch, latent_row_num, DEVICE='cuda'):
    inputs = torch.randn([1, args.latent_dim], device=DEVICE)
    t_w = g_ema.style(inputs).detach().unsqueeze(1)

    latent_t1 = t_w.repeat(batch, latent_row_num,1)
    return latent_t1


def cal_l1_loss(tensor1, tensor2):
    return torch.mean(torch.abs(tensor1 - tensor2))


def get_latent_mean_std(args, g_ema, DEVICE='cuda'):
    with torch.no_grad():
        n_mean_latent = 10000
        noise_sample = torch.randn(n_mean_latent, args.latent_dim, device=DEVICE)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
    return latent_mean, latent_std


def initial_tiny_noises(batch_size, noises_single):
    noises = []
    for noise in noises_single:
        # noises.append(noise.repeat(batch_size, 1, 1, 1))
        noises.append(noise.repeat(batch_size, 1, 1, 1).normal_().mul(0.001))
    for noise in noises:
        noise.requires_grad = True
    return noises


def own_noise(device = 'cuda'):

    noises = [torch.zeros(1, 1, 2 ** 2, 2 ** 2, device=device)]

    for i in range(3, 8 + 1):
        for _ in range(2):
            noises.append(torch.zeros(1, 1, 2 ** i, 2 ** i, device=device))

    return noises


def get_noises(g_ema, the_len): # The original module in the code
    noises_single = g_ema.make_noise()

    noises = []

    for noise in noises_single:
        noises.append(noise.repeat(the_len, 1, 1, 1).normal_())
    for noise in noises:
        noise.requires_grad = True
    return noises


def noise_regularize(noises):
    loss = 0
    for noise in noises:
        size = noise.shape[2]
        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8: break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss



def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def get_presaved_insp_latent():
    fpa = '/home/tech/Workspace/Lib/py3_lib/Other_ppl_based_models/Stylegan2/my_facelift_methods/ws_dstore/20_sampled_images_details.npy'
    cont_d = np.load(fpa, allow_pickle=True).item()
    return cont_d


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t
