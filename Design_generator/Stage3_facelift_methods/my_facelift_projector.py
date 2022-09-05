import os
import math
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import optim
import Design_generator.lpips as lpips
from torch.nn import _reduction as _Reduction
from Design_generator.tools.modified_stylegan import Generator
from Design_generator.my_utils import get_transform_fun, make_image


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (loss + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2))

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


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


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


def get_latent_mean_std(g_ema):
    with torch.no_grad():
        n_mean_latent = 10000
        noise_sample = torch.randn(n_mean_latent, 512, device=DEVICE)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
    return latent_mean, latent_std


def my_mse_loss(input, target, hl_mask):
    if target.requires_grad:
        ret = (input - target) ** 2
        ret = torch.mean(ret)
    else:
        input = (input * torch.tensor(hl_mask, device='cuda'))
        target = (target * torch.tensor(hl_mask, device='cuda'))

        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum('mean'))
    return ret


def my_percep_loss(generated_imgs, target, hl_mask):
    hl_mask = torch.tensor(hl_mask, device='cuda')
    hl_mask = hl_mask.unsqueeze(1).repeat(1, 3, 1, 1)

    generated_imgs = (generated_imgs * hl_mask)
    target = (target * hl_mask)
    lpips_loss = percept(generated_imgs, target).sum()
    mse = torch.mean((generated_imgs - target) ** 2)

    return lpips_loss, mse


def get_presaved_insp_latent():
    fpa = '/home/tech/Workspace/Lib/py3_lib/Other_ppl_based_models/Stylegan2/my_facelift_methods/ws_dstore/20_sampled_images_details.npy'
    cont_d = np.load(fpa, allow_pickle=True).item()
    return cont_d


def cal_l1_loss(tensor1, tensor2):
    return torch.mean(torch.abs(tensor1 - tensor2))


def get_random_latent(g_ema, args, batch, latent_row_num):
    global DEVICE
    inputs = torch.randn([batch, args.latent_dim], device=DEVICE)
    latent_t1 = g_ema.get_latent(inputs).detach()

    latent_t1 = latent_t1.unsqueeze(1).repeat(1, latent_row_num,1)
    return latent_t1


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


def save_rst(args, img_na_l, imgs, img_out_dir, epoch):

    for i, input_name in enumerate(img_na_l):
        img_name = f'{input_name}_lat-lambda_{args.latent_d_lambda}_epoch-{epoch}.jpg'
        Image.fromarray(imgs[i]).save(os.path.join(img_out_dir, img_name))


def read_mask(mask_fpa):
    img = Image.open(mask_fpa)
    mask = np.array(img.resize((256,256)))[:, :, -1]>100
    # mask = 1-enlarge_mask(mask, expend_size=25)
    # mask = 1 - mask
    mask = torch.tensor(mask)
    return mask


def read_saved_imgs(trans_func, img_dir, latent_dir, hl_mask_dir):

    cont_l = []
    for fna in os.listdir(latent_dir):
        t_key = fna.split('.')[0].replace('-project', '')
        img_fpa = os.path.join(img_dir, t_key+ '.jpg')

        latent_fpa = os.path.join(latent_dir, fna)
        hl_mask_fpa = os.path.join(hl_mask_dir, t_key+'.png')
        grill_mask = None

        t_tar_img = trans_func(Image.open(img_fpa).convert("RGB"))
        hl_mask = read_mask(hl_mask_fpa)

        t_latent = list(torch.load(latent_fpa).values())[0]['latent'].detach()
        cont_l.append((t_tar_img, hl_mask, grill_mask, t_latent, t_key))

    return cont_l


def check_rst_img(t_latent):
    img_gen, _ = g_ema(t_latent, input_is_latent=True, t_latent_n=14)
    Image.fromarray(make_image(img_gen)[0]).show()


def get_para():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='/home/tech/Workspace/Data/Projects_tmp/Facelift/style_GAN/210312/checkpoint/234000.pt')
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

    root_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/target_models'
    parser.add_argument("--root_dir", type=str, default=root_dir)
    parser.add_argument("--latent_dir", type=str,
                        default=os.path.join(root_dir, 'projected_uncong_latents_14'))
    parser.add_argument("--hl_mask_dir", type=str,
                        default=os.path.join(root_dir, 'masks/hl_mask'))
    parser.add_argument("--img_dir", type=str,
                        default=os.path.join(root_dir, 'processed_imgs'))
    parser.add_argument("--latent_out_dir", type=str,
                        default='/home/tech/Workspace/Data/Projects_working/Facelift/target_models/upgraded_rst/latent_new')
    parser.add_argument("--img_out_dir", type=str,
                        default='/home/tech/Workspace/Data/Projects_working/Facelift/target_models/upgraded_rst/facelift_rsts_end_5')

    return parser.parse_args()

def read_batch_cont(batch_l):
    imgs_l = [row[0] for row in batch_l]
    hl_mask_l = [row[1] for row in batch_l]
    grill_mask_l = [row[2] for row in batch_l]
    own_latent_l = [row[3] for row in batch_l]
    na_key_l = [row[4] for row in batch_l]

    imgs_npa = torch.stack(imgs_l, 0).to(DEVICE)
    hl_mask_npa = torch.stack(hl_mask_l, 0).to(DEVICE)
    own_latent_npa = torch.stack(own_latent_l, 0).to(DEVICE)

    return imgs_npa, hl_mask_npa, na_key_l, own_latent_npa


DEVICE = "cuda"
if __name__ == "__main__":
    args = get_para()

    for t_dir in [args.img_out_dir, args.latent_out_dir]:
        if not os.path.exists(t_dir):
            os.makedirs(t_dir)

    g_ema = init_generator(args)
    latent_mean, latent_std = get_latent_mean_std(g_ema)
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=DEVICE.startswith("cuda"))
    cont_l = read_saved_imgs(get_transform_fun(args.size), args.img_dir, args.latent_dir, args.hl_mask_dir) # !!! the content of the img
    already_exist_outputs_l = os.listdir(args.img_out_dir)
    z_start_idx, z_end_idx = 0, 5

    batch_size = 6
    for epoch in range(4):
        insp_latent = get_random_latent(g_ema, args, batch_size, args.t_latent_n)
        for args.lr in [0.005]: # or 0.01,
            for args.step in [300]:
                for args.latent_d_lambda in [4.8, 6.0, 7.2, 8.4]: # or 4.8
                    for idx in range(0, len(cont_l), batch_size):

                        imgs_npa, hl_mask_npa, na_key_l, own_latent_npa = read_batch_cont(cont_l[idx: idx+batch_size])
                        noises_single = g_ema.make_noise()
                        noises = []

                        for noise in noises_single:
                            noises.append(noise.repeat(imgs_npa.shape[0], 1, 1, 1).normal_())

                        # latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

                        #-- Clearify the gradient needs
                        own_latent_npa.requires_grad = True

                        for noise in noises:
                            noise.requires_grad = True
                        insp_latent.requires_grad = False
                        hl_mask_npa.requires_grad = False

                        optimizer = optim.Adam([own_latent_npa] + noises, lr=args.lr)

                        for i in tqdm(range(args.step), disable=False):
                            t = i / args.step
                            lr = get_lr(t, args.lr)
                            optimizer.param_groups[0]["lr"] = lr
                            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
                            latent_n = latent_noise(own_latent_npa, noise_strength.item())
                            img_gen = get_generated_imgs(g_ema, latent_n, noises)

                            n_loss = noise_regularize(noises)

                            # --------- Reginal facelit loss ---------
                            # p_loss = percept(img_gen, imgs_npa).sum()
                            # loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss


                            # --------- Reginal facelit loss ---------
                            latent_diff = cal_l1_loss(insp_latent[:, z_start_idx:z_end_idx],
                                                      own_latent_npa[:, z_start_idx:z_end_idx])
                            region_percp_loss, regional_mse_loss = my_percep_loss(img_gen, imgs_npa, hl_mask_npa)
                            loss = args.noise_regularize * n_loss + args.lpips_lambda * region_percp_loss + \
                                   args.mse_lambda * regional_mse_loss + args.latent_d_lambda* latent_diff

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            noise_normalize_(noises)

                        img_gen, _ = g_ema(own_latent_npa.detach().clone(), input_is_latent=True, noise=noises, t_latent_n=14)
                        imgs = make_image(img_gen)
                        # latent_fpa = os.path.join(latent_out_dir, na_key+'.pt')
                        # save_rst(args, na_key_l, imgs, args.img_out_dir, epoch)

