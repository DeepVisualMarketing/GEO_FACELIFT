import argparse
import os
import torch
from torch.nn import functional as F
import Design_generator.lpips as lpips
from Design_generator.my_utils import make_image
from Design_generator.tools.modified_stylegan import Generator
torch.manual_seed(17)


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


def init_generator(args):
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, args.latent_dim, 8).to('cuda')
    g.load_state_dict(ckpt['g_ema'])
    g.eval()
    return g


def get_paras():
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', default='w', choices=['z', 'w'])
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--n_sample', type=int, default=5000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--sampling', default='end', choices=['end', 'full'])
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--latent_dim', type=int, default=512)

    return parser.parse_args()


def get_modified_latents(args, batch, latent_t0):
    lat_row_num = latent_t0.shape[1]
    inputs = torch.randn([batch*lat_row_num, args.latent_dim], device=device)

    rand_lat = g.get_latent(inputs)
    latent_t1 = torch.stack([rand_lat[lat_row_num*i:lat_row_num*(i+1), :] for i in range(batch)], 0)

    lerp_t = torch.zeros(batch, device=device)

    latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None, None])
    latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None, None] + args.eps)
    latent_e = torch.stack([latent_e0, latent_e1], 1).view((latent_t0.shape[0]*2,*latent_t0.shape[1:]))
    return latent_e


if __name__ == '__main__':
    args = get_paras()
    device = 'cuda'
    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=device.startswith('cuda'))
    args.ckpt = f'/home/tech/Workspace/Data/Project_tmp/Facelift/style_GAN/210312/checkpoint/{177000}.pt'
    g = init_generator(args)


    for t_latent_n in [1, 2, 7, 14]:
        pic_num = 0
        projected_dir = f'/home/tech/Workspace/Data/Projects_working/Facelift/projection_rst/projected_uncong_latents_{t_latent_n}'
        batch_size = 16

        for args.eps in [0.5]:
            out_dir = f'/home/tech/Workspace/Data/Projects_working/Facelift/projection_rst/random_modified_imgs/lat-{t_latent_n}'
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            distances = []
            tar_l = os.listdir(projected_dir)

            for idx in range(0, len(tar_l), batch_size):
                proj_lat_l = []
                for fna in tar_l[idx:idx + batch_size]:
                    proj_fpa = os.path.join(projected_dir, fna)
                    proj_lat = list(torch.load(proj_fpa).items())[0][1]['latent']
                    proj_lat_l.append(proj_lat)

                proj_lats = torch.stack(proj_lat_l, 0).to(device)

                with torch.no_grad():
                    noise = g.make_noise()
                    latent_e = get_modified_latents(args, len(proj_lat_l), proj_lats)
                    image, _ = g(latent_e, input_is_latent=True, noise=noise, t_latent_n=t_latent_n)

                    if args.crop:
                        c = image.shape[2] // 8
                        image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

                    factor = image.shape[2] // 256

                    if factor > 1:
                        image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)

                    rst = make_image(image[1::2])
                    for i in range(rst.shape[0]):
                        # Image.fromarray(rst[i]).save(os.path.join(out_dir, f'{pic_num}.jpg'))
                        pic_num += 1


