import os
import math
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from torch import optim
import Design_generator.lpips as lpips
from torch.nn import functional as F
from Design_generator.my_utils import read_imgs, make_image
from Design_generator.tools.modified_stylegan import Generator


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


def get_generated_imgs(g_ema, latent_n, noises, t_latent_n):
    img_gen, _ = g_ema(latent_n, input_is_latent=True, noise=noises, t_latent_n=t_latent_n)

    batch, channel, height, width = img_gen.shape

    if height > 256:
        factor = height // 256

        img_gen = img_gen.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        img_gen = img_gen.mean([3, 5])
    return img_gen


def save_rst(img_na_l, noises, img_gen, latent_in, latent_out_dir, img_out_dir):
    imgs = make_image(img_gen)

    for i, input_name in enumerate(img_na_l):
        result_file = {}
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i : i + 1])

        result_file[input_name] = {
            "latent": latent_in[i],
            "noise": noise_single,
        }

        img_name = os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
        Image.fromarray(imgs[i]).save(os.path.join(img_out_dir, img_name))
        latent_fpa = os.path.join(latent_out_dir, img_name.replace('.png', '.pt'))
        torch.save(result_file, latent_fpa)


def find_opt(args, latent_std, latent_in, noises, g_ema):
    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)
    pbar = tqdm(range(args.step))
    latent_path = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        if not args.w_plus:
            latent_n = [latent_n]

        img_gen = get_generated_imgs(g_ema, latent_n, noises, args.t_latent_n)
        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)
        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss
        # loss = p_loss + args.noise_regularize * n_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"))
    return latent_path


def get_latent_mean_std(g_ema):
    global DEVICE
    n_mean_latent = 10000
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=DEVICE)
        latent_out = g_ema.style(noise_sample)
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
    return latent_mean, latent_std


def get_para():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='')
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.01) # pre 0.1
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--noise_regularize", type=float, default=1e5)
    parser.add_argument("--mse", type=float, default=0.8)
    parser.add_argument("--w_plus", type=bool, default=True)

    return parser.parse_args()


DEVICE = "cuda"

if __name__ == "__main__":

    args = get_para()
    for args.t_latent_n in [14]:
        args.ckpt = '/home/tech/Workspace/Data/Projects_tmp/Facelift/style_GAN/210312/checkpoint/234000.pt'
        tar_pic_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/target_models/previous_attempt/processed_imgs'
        latent_out_dir = f'/home/tech/Workspace/Data/Projects_working/Facelift/target_models/projected_uncong_latents_{args.t_latent_n}'
        img_out_dir = f'/home/tech/Workspace/Data/Projects_working/Facelift/target_models/projected_uncong_imgs_{args.t_latent_n}'

        for t_dir_pa in [img_out_dir, latent_out_dir]: os.makedirs(t_dir_pa, exist_ok=True)

        list_file_l, img_whole_sets = read_imgs(args.size, tar_pic_dir, img_out_dir, batch_size=8)
        g_ema = init_generator(args)
        latent_mean, latent_std = get_latent_mean_std(g_ema)
        percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=DEVICE.startswith("cuda"))
        noises_single = g_ema.make_noise()

        for round_idx, (img_na_l, imgs) in enumerate(zip(list_file_l, img_whole_sets)):

            imgs = torch.stack(imgs, 0).to(DEVICE)
            noises = []
            for noise in noises_single:
                noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
            latent_in = latent_in.unsqueeze(1).repeat(1, args.t_latent_n, 1)

            latent_in.requires_grad = True
            for noise in noises:
                noise.requires_grad = True

            latent_path = find_opt(args, latent_std, latent_in, noises, g_ema)
            last_latent = [latent_path[-1]] if not args.w_plus else latent_path[-1]
            img_gen, _ = g_ema(last_latent, input_is_latent=True, noise=noises, t_latent_n=args.t_latent_n)
            Image.fromarray(make_image(img_gen)[0]).show()
            # save_rst(img_na_l, noises, img_gen, latent_in, latent_out_dir, img_out_dir)
