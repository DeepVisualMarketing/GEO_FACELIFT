import os
import sys
import torch
import argparse
from PIL import Image
from Design_generator.tools.modified_stylegan import Generator
import numpy as np
from own_pathes import own_path_d
sys.path.append(own_path_d['Pic'])
from Design_generator.my_utils import get_transform_fun, make_image
from random import shuffle


def black_edges(t_img):
    img_npa = np.array(t_img)
    for y in range(img_npa.shape[0]):
        for x in range(img_npa.shape[1]):
            if x in [0,255] or y in [0,255]:
                img_npa[y,x] = (0,0,0)

    return Image.fromarray(img_npa)


def generate_compare_matirx(cont_d, out_fpa):

    out_image = Image.new("RGB", (256 * 7, 256 * 7), color=(255,255,255))

    for (start_idx, end_idx) in cont_d:
        t_img = black_edges(cont_d[(start_idx, end_idx)])
        t_img = cont_d[(start_idx, end_idx)]
        # out_image.paste(t_img, (start_idx*256, end_idx*256))
        t_img.save(os.path.join(f'ws_dstore/{start_idx+1}-{end_idx+1}.png'))

    out_image.save(out_fpa)

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

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def init_generator(args):
    global DEVICE
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(DEVICE)
    return g_ema


def get_generated_imgs(g_ema, latent_n, noises):
    img_gen, _ = g_ema(latent_n, input_is_latent=True, noise=noises, t_latent_n=14)

    batch, channel, height, width = img_gen.shape

    if height > 256:
        factor = height // 256

        img_gen = img_gen.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        img_gen = img_gen.mean([3, 5])
    return img_gen


def check_rst_img(t_latent):
    img_gen, _ = g_ema(t_latent, input_is_latent=True, t_latent_n=14)
    Image.fromarray(make_image(img_gen)[0]).show()


def get_para():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='/home/tech/Workspace/Data/Project_tmp/Facelift/style_GAN/210312/checkpoint/177000.pt')
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default = 200)
    parser.add_argument("--noise_regularize", type=float, default=1e4)
    parser.add_argument("--mse", type=float, default=1)
    parser.add_argument("--w_plus", action="store_true")
    parser.add_argument("--t_latent_n", type=int, default=14)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--latent_d_lambda", type=float, default=4.8)
    parser.add_argument("--lpips_lambda", type=float, default=0.4)
    parser.add_argument("--mse_lambda", type=float, default=10.0)

    return parser.parse_args()

def get_cand_color_latent(insp_latent_dir, color):
    cand_l = os.listdir(insp_latent_dir)
    shuffle(cand_l)
    rst_l = []
    for fna in cand_l:
        if fna.split('$$')[3] != color and 'Toyota$$Yaris$$2014' in fna:
            rst_l.append(os.path.join(insp_latent_dir, fna))
            if len(rst_l) >= 80:
                return rst_l

    return rst_l


DEVICE = "cuda"
if __name__ == "__main__":
    root_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/target_models'
    tar_model_latent_dir = os.path.join(root_dir, 'projected_uncong_latents_14')
    insp_latent_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/projection_rst/projected_uncong_latents_14'
    img_out_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/target_models/upgraded_rst/original_style_mixing'
    args = get_para()
    if not os.path.exists(img_out_dir): os.makedirs(img_out_dir)

    g_ema = init_generator(args)

    for fna in os.listdir(tar_model_latent_dir):
        if 'Vauxhall' not in fna: continue

        print('start', fna)
        latent_fpa = os.path.join(tar_model_latent_dir, fna)
        color = fna.split('$$')[3]
        if color != 'White': continue

        t_latent = list(torch.load(latent_fpa).values())[0]['latent'].detach()
        cand_latent_l = get_cand_color_latent(insp_latent_dir, color)

        for insp_fpa in cand_latent_l:
            if 'Toyota$$Yaris$$2014$$Black$$876$$image_8-project' not in insp_fpa: continue
            t_insp_latent = list(torch.load(insp_fpa).values())[0]['latent'].detach()
            tmp_d = {}
            for z_start_idx in [0,1,3,5,7,9,11]:
                for z_end_idx in [1, 3,5,7,9,11, 14]:
                    if z_end_idx <= z_start_idx: continue
                    t_copy = t_latent.clone()
                    t_copy[z_start_idx:z_end_idx] = t_insp_latent[z_start_idx:z_end_idx]


                    img_gen, _ = g_ema(t_copy.unsqueeze(0).to(DEVICE), input_is_latent=True,t_latent_n=14)
                    imgs = make_image(img_gen)
                    tmp_idx_func = lambda x: [0, 1, 3, 5, 7, 9, 11, 14].index(x)
                    tmp_d[(tmp_idx_func(z_start_idx), tmp_idx_func(z_end_idx)-1)] = Image.fromarray(imgs[0])

            generate_compare_matirx(tmp_d, os.path.join(img_out_dir, fna.replace('project.pt', f"{insp_fpa.split('/')[-1].split('.')[0]}.jpg")))
