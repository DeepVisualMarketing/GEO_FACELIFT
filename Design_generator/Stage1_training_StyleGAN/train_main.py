import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import argparse
from tqdm import tqdm
from torch import nn, optim
from torch.utils import data
from torchvision import transforms, utils
from Design_generator.tools.dataset import MultiResolutionDataset
from Design_generator.tools.original_model import Generator, Discriminator
from Design_generator.tools.style_distributed import (get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size,)
from Design_generator.Stage1_training_StyleGAN.training_utils import (sample_data, accumulate, data_sampler, requires_grad, mixing_noise,
                            d_logistic_loss, g_nonsaturating_loss, d_r1_loss, g_path_regularize)


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    d_loss_val, g_loss_val = 0, 0
    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}


    g_module = generator
    d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader).to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        #-- The ranbdom latent inputs
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)


        fake_img, _ = generator(noise)
        real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    utils.save_image(sample, f"{args.rst_dir}/sample/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5), normalize=True, range=(-1, 1),)

            if i % 2500 == 0:
                torch.save({
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,},
                    f"{args.rst_dir}/checkpoint/{str(i).zfill(6)}.pt",)


def initial_folders(args):
    for folder in ['sample', 'checkpoint']:
        os.makedirs(os.path.join(args.rst_dir, folder), exist_ok=True)


def get_default_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='/home/tech/Workspace/Data/Projects_working/Facelift/data/large_one/style_lmdb')
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    # parser.add_argument("--ckpt", type=str, default='/home/tech/Workspace/Data/Project_tmp/Facelift/style_GAN/210826/checkpoint/132500.pt')
    parser.add_argument("--ckpt", type=str,
                        default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--rst_dir", type=str, default='/home/tech/Workspace/Data/Projects_tmp/Facelift/style_GAN/220831')


    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda"
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args = get_default_setting()
    initial_folders(args)
    args.latent = 2048
    args.n_mlp = 3

    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),)
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),)

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        ckpt_name = os.path.basename(args.ckpt)
        args.start_iter = int(os.path.splitext(ckpt_name)[0]) + 1

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),])

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(dataset, batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=False), drop_last=True,)

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
