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


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.sample, args.latent, device=device)

           sample, latent = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, return_latents=True)
           
           utils.save_image(
            sample,
            f'sample/{str(i).zfill(6)}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )


def generate_individual(args, g_ema, mean_latent, latent_d, inject_index, out_dir):
    with torch.no_grad():
        g_ema.eval()
        old_img_l = []

        for row, old_model in enumerate(sorted(latent_d['target'].keys())):
            old_sample, _ = g_ema([latent_d['target'][old_model].unsqueeze(0)], truncation=args.truncation,
                                  truncation_latent=mean_latent,
                                  return_latents=False, input_is_latent=True)
            old_img = utils.get_image(old_sample, nrow=1, normalize=True, range=(-1, 1))


            old_img_l.append(old_img)
            for col, new_model in enumerate(sorted(latent_d['low'].keys())):
                generated_sample, _ = g_ema([latent_d['target'][old_model].unsqueeze(0),
                                             latent_d['low'][new_model].unsqueeze(0)],
                                            truncation=args.truncation, truncation_latent=mean_latent,
                                       return_latents=True, input_is_latent=True, inject_index=inject_index)

                mid_img = utils.get_image(generated_sample, nrow=1, normalize=True, range=(-1, 1))
                out_fpa = os.path.join(out_dir, "{}_{}.png".format(old_model,new_model))
                mid_img.save(out_fpa)

def generate_pair(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z1 = torch.randn(args.sample, args.latent, device=device)
            sample1, latent1 = g_ema([sample_z1], truncation=args.truncation, truncation_latent=mean_latent,
                                   return_latents=True)
            sample_z2 = torch.randn(args.sample, args.latent, device=device)
            sample2, latent2 = g_ema([sample_z2], truncation=args.truncation, truncation_latent=mean_latent,
                                   return_latents=True)

            sample_mid, latent_mid = g_ema([latent1[:,0,:], latent2[:,0,:]], truncation=args.truncation, truncation_latent=mean_latent,
                                   return_latents=True, input_is_latent=True)

            imgs_l = []
            for t_sample in [sample1, sample2, sample_mid]:
                imgs_l.append(utils.get_image(t_sample, nrow=1, normalize=True, range=(-1, 1)))

            out_image  = Image.new("RGBA", ( 256*3, 256) )
            for z in range(3):
                out_image.paste(imgs_l[z], (z*256,0))
                out_image.save("round_{}.png".format(i))


def generate_compare_matirx(args, inject_index, g_ema, mean_latent, latent_d, out_dir):
    with torch.no_grad():
        g_ema.eval()
        out_image = Image.new("RGBA", (256 * 6, 256 * 6), color=(255,255,255))
        old_img_l = []
        new_img_l = []

        for row, old_model in enumerate(sorted(latent_d['old'].keys())):
            old_sample, _ = g_ema([latent_d['old'][old_model].unsqueeze(0)], truncation=args.truncation,
                                  truncation_latent=mean_latent,
                                  return_latents=False, input_is_latent=True)
            old_img = utils.get_image(old_sample, nrow=1, normalize=True, range=(-1, 1))
            out_image.paste(old_img, (0, (row+1)*256))
            old_img_l.append(old_img)
            for col, new_model in enumerate(sorted(latent_d['new'].keys())):
                new_sample, _ = g_ema([latent_d['new'][new_model].unsqueeze(0)], truncation=args.truncation, truncation_latent=mean_latent,
                                       return_latents=False, input_is_latent=True)
                if row == 0:
                    new_img = utils.get_image(new_sample, nrow=1, normalize=True, range=(-1, 1))
                    out_image.paste(new_img, ((col + 1) * 256, 0))
                    new_img_l.append(new_img)

                generated_sample, _ = g_ema([latent_d['old'][old_model].unsqueeze(0), latent_d['new'][new_model].unsqueeze(0)], truncation=args.truncation, truncation_latent=mean_latent,
                                       return_latents=True, input_is_latent=True, inject_index=inject_index)

                mid_img = utils.get_image(generated_sample, nrow=1, normalize=True, range=(-1, 1))
                out_image.paste(mid_img, ((col + 1) * 256, (row+1)*256))

                # small_out_image = Image.new("RGBA", (256 * 4, 256), color=(255,255,255))
                # for z, t_img in enumerate([old_img_l[row], new_img_l[col], mid_img], ):
                #     small_out_image.paste(t_img, (int(z* 256*4/3), 0))
                #
                # small_out_image.save(os.path.join(out_dir, "{}_{}.png".format(old_model, new_model)))

                # mid_img.save(os.path.join(out_dir,"{}_{}_{}.png".format(old_model, new_model, inject_index)))

        out_image.save(os.path.join(out_dir, "facelift_ij{}.png".format(inject_index)))

def read_latents(args):
    saved_d = torch.load(args.saved_latent_pa)
    rst_d = {'old':{}, 'new':{}}
    for t_fpa in saved_d:
        fna = t_fpa.split('/')[-1]
        model = '$$'.join(fna.split('$$')[:2])
        if 'A1' in model or 'Focus' in model or 'Astra' in model:
            continue

        saved_latent = saved_d[t_fpa]['latent']


        if int(fna.split('$$')[2]) > 2013:
            rst_d['new'][model] = saved_latent
        else:
            rst_d['old'][model] = saved_latent
    return rst_d

def read_latents2(args):
    rst_d = {'high': {}, 'low': {}, 'target':{}}
    for i in range(1):
        pt_fpa = '/home/tech/Workspace/Projects/Facelift/stage_II/stylegan2/projector/low_stylish_insp_{}.pt'.format(i)
        saved_d = torch.load(pt_fpa)

        for t_fpa in saved_d:
            fna = t_fpa.split('/')[-1]
            model = '$$'.join(fna.split('$$')[:2])
            saved_latent = saved_d[t_fpa]['latent']
            if fna[:2] == 'h_':
                rst_d['high'][fna] = saved_latent
            elif fna[:2] == 'l_':
                rst_d['low'][fna] = saved_latent
            else:
                rst_d['target'][fna] = saved_latent
    return rst_d


if __name__ == '__main__':
    device = 'cuda'
    args = read_setting()
    latent_d = read_latents2(args)
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

    out_dir = '/home/tech/Workspace/Data/Project_tmp/Facelift/stage_II_exp/stylegan_10_06/setting_5_to_end_for_low'

    for inject_index in range(10, 11):
        generate_individual(args, g_ema, mean_latent, latent_d, inject_index, out_dir)
