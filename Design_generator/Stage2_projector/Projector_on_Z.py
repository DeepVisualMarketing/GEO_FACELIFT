import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Design_generator.lpips import PerceptualLoss
from Design_generator.tools.MyPICDataset import MyPICDataset
from Design_generator.tools.modified_stylegan import Generator
from Design_generator.my_utils import get_transform_fun, make_image
from Design_generator.tools.original_model import EqualLinear, PixelNorm
from Design_generator.tools.my_modules import initial_tiny_noises, own_noise, get_lr



def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def rename_para(t_l, idx_minus=False):
    new_l = []
    for idx, (key, val) in enumerate(t_l):
        new_idx = idx//2 if idx_minus else idx//2+1
        key = f'{new_idx}.{key.split(".")[-1]}'
        new_l.append((key, val))

    return dict(new_l)


def recreate_mapping(args, split_idx, saved_d):
    split_mapping_p1 = [PixelNorm()]
    split_mapping_p2 = []

    for i in range(args.n_mlp):
        if i < split_idx:
            split_mapping_p1.append(EqualLinear(args.latent_dim, args.latent_dim, lr_mul=0.01, activation='fused_lrelu'))
        else:
            split_mapping_p2.append(EqualLinear(args.latent_dim, args.latent_dim, lr_mul=0.01, activation='fused_lrelu'))

    mapping_p1 = nn.Sequential(*split_mapping_p1)
    mapping_p2 = nn.Sequential(*split_mapping_p2)
    t_processed_d1 = rename_para(list(saved_d.items())[:split_idx * 2])
    t_processed_d2 = rename_para(list(saved_d.items())[split_idx * 2:], idx_minus=True)

    mapping_p1.load_state_dict(t_processed_d1)
    mapping_p2.load_state_dict(t_processed_d2)
    return mapping_p1, mapping_p2


def init_generator(args, mapping_between_idx):
    global DEVICE
    g_ema = Generator(args.img_size, args.latent_dim, args.n_mlp)
    t_saved_status_d = torch.load(args.ckpt)["g_ema"]

    style_d = dict([(key,val) for key,val in t_saved_status_d.items() if 'style' in key])
    g_ema.load_state_dict(t_saved_status_d, strict=False)
    mapping_p1, mapping_p2 = recreate_mapping(args, mapping_between_idx, style_d)
    g_ema.style = mapping_p2
    g_ema.eval()
    g_ema = g_ema.to(DEVICE)
    mapping_p1.cuda()
    return mapping_p1, g_ema


def get_presaved_insp_latent():
    fpa = '/home/tech/Workspace/Lib/py3_lib/Other_ppl_based_models/Stylegan2/my_facelift_methods/ws_dstore/20_sampled_images_details.npy'
    cont_d = np.load(fpa, allow_pickle=True).item()
    return cont_d


def get_random_latent(g_ema, args, batch, latent_row_num):
    global DEVICE
    inputs = torch.randn([batch, args.latent_dim], device=DEVICE)
    latent_t1 = g_ema.get_latent(inputs).detach()

    latent_t1 = latent_t1.unsqueeze(1).repeat(1, latent_row_num,1)
    return latent_t1


def save_rst(fna_l, imgs, img_out_dir):
    for idx, fna in enumerate(fna_l):
        Image.fromarray(imgs[idx]).save(os.path.join(img_out_dir, fna))


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


def my_get_mean_latent(args, tmp_model):
    input = torch.randn(10000, args.latent_dim, device=DEVICE)
    output = tmp_model(input)
    return torch.mean(output,0).detach()


def get_para():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.05) # 0.05
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default = 500)
    parser.add_argument("--noise_regularize", type=float, default=1e4)  #1e4
    parser.add_argument("--mse", type=float, default=0.8)
    parser.add_argument("--w_plus", action="store_true")
    parser.add_argument("--t_latent_n", type=int, default=14)
    parser.add_argument("--latent_d_lambda", type=float, default=4.8)
    parser.add_argument("--lpips_lambda", type=float, default=0.4)
    parser.add_argument("--mse_lambda", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--img_out_dir", type=str,
                        default='')
    parser.add_argument("--tar_dir", type=str,
                        default='/home/tech/Workspace/Data/Projects_working/Facelift/data/large_one/StyleGAN_data_test')

    args = parser.parse_args()
    args.ckpt = '/home/tech/Workspace/Data/Projects_tmp/Facelift/style_GAN/210826/checkpoint/132500.pt'
    args.latent_dim = 2048
    args.n_mlp = 3

    return args, 'cuda'


if __name__ == "__main__":
    args, DEVICE = get_para()
    percept = PerceptualLoss(model="net-lin", net="vgg", use_gpu=DEVICE.startswith("cuda"))

    for mapping_between_idx in [2]:
        for num_of_mapping in [1, 7, 14]:
            args.img_out_dir = f'/home/tech/Workspace/Data/Data_disk2/research/code_data/Facelift_generator_2(2048_3)/projection_compare/by_{num_of_mapping}z_middle_mapping{mapping_between_idx}'
            if not os.path.exists(args.img_out_dir): os.makedirs(args.img_out_dir)

            my_dataset = MyPICDataset([args.tar_dir], exist_l=os.listdir(args.img_out_dir), transform=get_transform_fun(args.img_size))
            data_loader = DataLoader(my_dataset, shuffle=True,
                                     batch_size=args.batch_size)  # , sampler=torch.utils.data.RandomSampler()
            pre_mapping_front, g_ema = init_generator(args, mapping_between_idx)
            new_mean = my_get_mean_latent(args, pre_mapping_front)

            for imgs_npa, fna_l in data_loader:
                if len(os.listdir(args.img_out_dir)) > 200: break

                # own_latent_npa = latent_mean.detach().clone().repeat(args.batch_size, 1).unsqueeze(1).to(DEVICE)
                noises = initial_tiny_noises(args.batch_size, own_noise())
                rst_d = {}
                sample_z_l = []

                for i in range(num_of_mapping):
                    sample_z_l.append(new_mean.unsqueeze(0).repeat(args.batch_size,1))
                    sample_z_l[-1].requires_grad = True

                optimizer = optim.Adam(sample_z_l, lr=args.lr, betas=(0.5, 0.99))
                for i in tqdm(range(args.step), disable=False):

                    sample_z_based_w = torch.cat([g_ema.style(sample_z).unsqueeze(1) for sample_z in sample_z_l], 1)

                    lr = get_lr(i / args.step, args.lr)
                    optimizer.param_groups[0]["lr"] = lr

                    img_gen, _ = g_ema(sample_z_based_w, input_is_latent=True, noise=noises, t_latent_n=num_of_mapping)

                    # --------- Reginal facelit loss ---------
                    p_loss = percept(img_gen, imgs_npa).sum()
                    mse_loss = F.mse_loss(img_gen, imgs_npa.to(DEVICE))
                    loss = p_loss + args.mse * mse_loss
                    if i %25 ==0:
                        print(i, ' loss', p_loss.data/args.batch_size)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                img_gen, _ = g_ema(sample_z_based_w.detach().clone(), input_is_latent=True, noise=noises, t_latent_n=num_of_mapping)
                save_rst(fna_l, make_image(img_gen), args.img_out_dir)
