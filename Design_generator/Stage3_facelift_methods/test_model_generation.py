import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torchvision import utils
from Design_generator.tools.modified_stylegan import Generator
import argparse


def generate_individual(args, g_ema, mean_latent, t_latent, out_dir):
    with torch.no_grad():
        g_ema.eval()
        t_latent_n = 14
        t_sample, _ = g_ema(t_latent.repeat(1, 14, 1), t_latent_n, truncation=args.truncation,
                              truncation_latent=mean_latent,
                              return_latents=False, input_is_latent=True)

        mid_img = utils.get_image(t_sample, nrow=1, normalize=True, range=(-1, 1))
        out_fpa = os.path.join(out_dir, "t_test.png")


def read_setting():
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="/home/tech/Workspace/Data/Project_tmp/Facelift/style_GAN/210312/checkpoint/177000.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8
    return args


if __name__ == '__main__':
    device = 'cuda'
    args = read_setting()

    saved_latent_fpa = '/home/tech/Workspace/Projects/Facelift/stylegan2/projector/low_stylish_insp_0.pt'
    latent_d = torch.load(saved_latent_fpa)
    one_latent = latent_d[list(latent_d.keys())[0]]['latent']

    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    out_dir = '/home/tech/Workspace/Lib/py3_lib/Other_ppl_based_models/Stylegan2/my_test_modules/tmp'
    generate_individual(args, g_ema, mean_latent, one_latent, out_dir)
