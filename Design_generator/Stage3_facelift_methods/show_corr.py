import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import utils
from Design_generator.tools.modified_stylegan import Generator
from Design_generator.tools.pic_utils import npa_to_image


def generate_individual(args, g_ema, mean_latent, t_latent, out_dir):
    with torch.no_grad():
        g_ema.eval()
        t_latent_n = 14
        t_sample, _ = g_ema(t_latent.repeat(1, 14, 1), t_latent_n, truncation=args.truncation,
                              truncation_latent=mean_latent,
                              return_latents=False, input_is_latent=True)
        # The input t_latent has shape [1, 14, 512]
        mid_img = utils.get_image(t_sample, nrow=1, normalize=True, range=(-1, 1))
        out_fpa = os.path.join(out_dir, "t_test.png")
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


if __name__ == '__main__':
    device = 'cuda'
    args = read_setting()

    saved_latent_fpa = '/home/tech/Workspace/Projects/Facelift/stylegan2/projector/low_stylish_insp_0.pt'
    latent_d = torch.load(saved_latent_fpa)
    one_latent = latent_d[list(latent_d.keys())[0]]['latent']

    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])
    img_npa = np.zeros((14,512))
    t_first_weight = g_ema.conv1.conv.modulation.weight.detach().cpu().numpy()

    for i in range(8,13):
        t_first_weight = g_ema.convs[i].conv.modulation.weight.detach().cpu().numpy()
        img = npa_to_image(t_first_weight)
        Image.fromarray(img).show()



