import argparse
import os
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import lpips
from modified_stylegan import Generator
import time
torch.manual_seed(17)


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


def init_generator(args):
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, args.latent_dim, 8).to('cuda')
    g.load_state_dict(ckpt['g_ema'])
    g.eval()
    return g


def get_paras():
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', default='w', choices=['z', 'w'])
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--n_sample', type=int, default=5000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--sampling', default='end', choices=['end', 'full'])
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--latent_dim', type=int, default=512)

    return parser.parse_args()


def get_modified_latents(args, batch, latent_t0):

    inputs = torch.randn([batch, args.latent_dim], device=device)
    latent_t1 = g.get_latent(inputs)

    lerp_t = torch.zeros(batch, device=device)
    latent_t1 = latent_t1.unsqueeze(1).repeat(1,latent_t0.shape[1],1)

    latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None, None])
    latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None, None] + args.eps)
    latent_e = torch.stack([latent_e0, latent_e1], 1).view((latent_t0.shape[0]*2,*latent_t0.shape[1:]))
    return latent_e


if __name__ == '__main__':
    args = get_paras()
    device = 'cuda'
    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda'))
    args.ckpt = f'/home/tech/Workspace/Data/Project_tmp/Facelift/style_GAN/210312/checkpoint/{177000}.pt'
    g = init_generator(args)

    for t_latent_n in [1, 2, 7, 14]:
        projected_dir = f'/home/tech/Workspace/Data/Projects_working/Facelift/projection_rst/projected_uncong_latents_{t_latent_n}'
        tar_l = os.listdir(projected_dir)
        batch_size = 16
        distances = []

        for idx in range(0, len(tar_l), batch_size):
            print(idx)
            proj_lat_l = []
            for fna in tar_l[idx:idx + batch_size]:
                proj_fpa = os.path.join(projected_dir, fna)
                proj_lat = list(torch.load(proj_fpa).items())[0][1]['latent']
                proj_lat_l.append(proj_lat)

            proj_lats = torch.stack(proj_lat_l, 0).to(device)


            with torch.no_grad():
                noise = g.make_noise()
                latent_e = get_modified_latents(args, len(proj_lat_l), proj_lats)
                image, _ = g(latent_e, input_is_latent=True, noise=noise, t_latent_n=t_latent_n)

                if args.crop:
                    c = image.shape[2] // 8
                    image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

                factor = image.shape[2] // 256

                if factor > 1:
                    image = F.interpolate(
                        image, size=(256, 256), mode='bilinear', align_corners=False
                    )

                dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                    args.eps ** 2
                )
                distances.append(dist.to('cpu').numpy())

        distances = np.concatenate(distances, 0)

            # lo = np.percentile(distances, 1, interpolation='lower')
            # hi = np.percentile(distances, 99, interpolation='higher')
            # filtered_dist = np.extract(
            #     np.logical_and(lo <= distances, distances <= hi), distances
            # )

        with open('ws_dstore/ppl_of_t_latent_n.csv', 'a') as f_out:
            f_out.write(f'{t_latent_n},{distances.mean()}\n')
