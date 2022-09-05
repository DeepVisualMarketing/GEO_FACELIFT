import torch
import argparse
import numpy as np
import Design_generator.lpips as lpips
from Design_generator.tools.original_model import Generator


def combine_w_latent(w1_latent, w2_latent, inj_idx):
    new_latent = w1_latent.detach().clone().repeat(1, 14, 1)
    new_latent[0, inj_idx, :] = w2_latent.detach().clone()
    return new_latent


def generate_img(args, g_ema, mean_latent, t_latent):
    with torch.no_grad():
        g_ema.eval()
        t_sample, _ = g_ema(t_latent.repeat(1, 14, 1), truncation=args.truncation, truncation_latent=mean_latent,
                              return_latents=False, input_is_latent=True) # The input t_latent has shape [1, 14, 512]

        return t_sample


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

    DEVICE = "cuda"
    saved_fpa = '/home/tech/Workspace/Lib/py3_lib/Other_ppl_based_models/Stylegan2/my_test_modules/ws_dstore/1000_w_latent_sample.npy'
    latent_l = np.load(saved_fpa, allow_pickle=True)
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=DEVICE.startswith("cuda"))

    rst_d = {}
    for i in range(14):
        rst_d[i] = []

    for t_idx, one_latent in enumerate(latent_l):
        print(t_idx)
        compare_latent = latent_l[int(np.random.uniform() * 1000)]
        for inj_idx in range(14):
            g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
            g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])

            if args.truncation < 1:
                with torch.no_grad():
                    mean_latent = g_ema.mean_latent(args.truncation_mean)
            else:
                mean_latent = None

            combined_latent = combine_w_latent(one_latent, compare_latent, inj_idx)

            original_img = generate_img(args, g_ema, mean_latent, one_latent)
            modified_img = generate_img(args, g_ema, mean_latent, combined_latent)
            the_diff = percept(original_img, modified_img, retPerLayer=True)
            the_diff = [ele.detach().cpu().numpy().item() for ele in the_diff[1]]
            rst_d[inj_idx].append(the_diff)


# np.save('ws_dstore/the_diff_vals', rst_d)
