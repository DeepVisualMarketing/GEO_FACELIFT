import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from read_default_setting import read_setting
import numpy as np
from PIL import Image
import os


def generate_compare_matirx(args, inject_index, g_ema, mean_latent, latent_d, out_dir):
    with torch.no_grad():
        g_ema.eval()
        out_image = Image.new("RGBA", (256 * 6, 256 * 4), color=(255,255,255))
        tar_band = 'Fiesta'
        old_img_l = []
        new_img_l = []
        tar_l = [key for key in sorted(latent_d['target'].keys()) if tar_band in key]
        for row, old_model in enumerate(tar_l):
            old_sample, _ = g_ema([latent_d['target'][old_model].unsqueeze(0)], truncation=args.truncation,
                                  truncation_latent=mean_latent,
                                  return_latents=False, input_is_latent=True)
            old_img = utils.get_image(old_sample, nrow=1, normalize=True, range=(-1, 1))
            out_image.paste(old_img, (0, (row+1)*256))
            old_img_l.append(old_img)
            for col, new_model in enumerate(sorted(latent_d['high'].keys())):
                new_sample, _ = g_ema([latent_d['high'][new_model].unsqueeze(0)], truncation=args.truncation, truncation_latent=mean_latent,
                                       return_latents=False, input_is_latent=True)
                if row == 0:
                    new_img = utils.get_image(new_sample, nrow=1, normalize=True, range=(-1, 1))
                    out_image.paste(new_img, ((col + 1) * 256, 0))
                    new_img_l.append(new_img)

                generated_sample, _ = g_ema([latent_d['target'][old_model].unsqueeze(0), latent_d['high'][new_model].unsqueeze(0)], truncation=args.truncation, truncation_latent=mean_latent,
                                       return_latents=True, input_is_latent=True, inject_index=inject_index)

                mid_img = utils.get_image(generated_sample, nrow=1, normalize=True, range=(-1, 1))
                out_image.paste(mid_img, ((col + 1) * 256, (row+1)*256))

                # small_out_image = Image.new("RGBA", (256 * 4, 256), color=(255,255,255))
                # for z, t_img in enumerate([old_img_l[row], new_img_l[col], mid_img], ):
                #     small_out_image.paste(t_img, (int(z* 256*4/3), 0))
                #
                # small_out_image.save(os.path.join(out_dir, "{}_{}.png".format(old_model, new_model)))

                # mid_img.save(os.path.join(out_dir,"{}_{}_{}.png".format(old_model, new_model, inject_index)))

        out_image.save(os.path.join(out_dir, f"{tar_band}_facelift_ij{inject_index}.png"))


def read_latents():
    rst_d = {'high': {}, 'low': {}, 'target': {}}

    latent_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/target_models/t_latents'
    for fna in os.listdir(latent_dir):
        t_d = torch.load(os.path.join(latent_dir, fna))
        rst_d['target'][fna.split('.')[0]] = t_d[list(t_d.keys())[0]]['latent']

    latent_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/inspiring_designs/high_stylish/t_latents'
    for fna in os.listdir(latent_dir):
        t_d = torch.load(os.path.join(latent_dir, fna))
        rst_d['high'][fna.split('.')[0]] = t_d[list(t_d.keys())[0]]['latent']

    return rst_d


if __name__ == '__main__':
    device = 'cuda'
    args = read_setting()
    latent_d = read_latents()
    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    out_dir = '/home/tech/Workspace/Data/Project_tmp/Facelift/target_models/mix_with_high_style'

    for inject_index in range(4, 11):
        generate_compare_matirx(args, inject_index, g_ema, mean_latent, latent_d, out_dir)
