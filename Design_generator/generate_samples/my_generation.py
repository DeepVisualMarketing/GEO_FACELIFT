import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse


def generate_individual(args, g_ema, mean_latent, t_latent, out_fpa, insp_latent=None, inj_place=None):
    with torch.no_grad():
        if insp_latent is not None:
            t_latent[inj_place:] = insp_latent.repeat(1, 14-inj_place, 1)

        g_ema.eval()
        t_sample, _ = g_ema(t_latent.unsqueeze(0), truncation=args.truncation,
                              truncation_latent=mean_latent,
                              return_latents=False, input_is_latent=True)
        # The input t_latent has shape [1, 14, 512]

        mid_img = utils.get_image(t_sample, nrow=1, normalize=True, range=(-1, 1))
        mid_img.save(out_fpa)


def read_setting():
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="/home/tech/Workspace/Data/Project_tmp/Facelift/style_GAN/10-06/checkpoint/250000.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8
    return args


def read_latents(tar_latend_dir, insp_latend_dir):

    insp_latend_d, tar_latent_d = {}, {}
    for fna in os.listdir(insp_latend_dir):
        fpa = os.path.join(insp_latend_dir, fna)
        t_latent = list(torch.load(fpa).values())[0]['latent']
        insp_latend_d['$$'.join(fna.split('$$')[:3])] = t_latent

    for fna in os.listdir(tar_latend_dir):
        fpa = os.path.join(tar_latend_dir, fna)
        t_latent = list(torch.load(fpa).values())[0]['latent']
        tar_latent_d['$$'.join(fna.split('$$')[:3])] = t_latent

    return tar_latent_d, insp_latend_d



if __name__ == '__main__':
    device = 'cuda'
    args = read_setting()

    insp_latend_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/inspiring_designs/new_210210/high/t_latents'
    tar_latend_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/target_models/t_latents'
    out_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/inspiring_designs/new_210210/high/mix_with_target_designs_inj8'
    os.makedirs(out_dir)

    tar_latent_d, insp_latend_d = read_latents(tar_latend_dir, insp_latend_dir)

    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    for tar_design in tar_latent_d:
        for insp_design in insp_latend_d:
            out_fpa = os.path.join(out_dir, f'{tar_design}_mix_{insp_design}.png')
            generate_individual(args, g_ema, mean_latent, tar_latent_d[tar_design], out_fpa, insp_latent=insp_latend_d[insp_design], inj_place=8)
