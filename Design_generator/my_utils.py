import torch
from torchvision import transforms
import os
from PIL import Image
from Design_generator.tools.calc_inception import load_patched_inception_v3
import numpy as np
from tqdm import tqdm
from scipy import linalg
from torch import nn
from Design_generator.tools.my_modules import lerp


def read_mask(mask_fpa, tar_size = 256):
    img = Image.open(mask_fpa)
    mask_raw_npa = np.array(img.resize((tar_size, tar_size)))

    mask = mask_raw_npa[:, :, -1]>100 if len(mask_raw_npa.shape)>2 else mask_raw_npa[:, :]>150

    # mask = 1-enlarge_mask(mask, expend_size=25)
    # mask = 1 - mask
    mask = torch.tensor(mask)
    return mask


def read_facelift_cand_dirs(trans_func, img_dir, hl_mask_dir, RETURN_EARLY):
    cont_l = []
    img_na_l = [ele for ele in os.listdir(img_dir) if ele[-3:] == 'jpg']

    for fna in img_na_l:
        fna_key = fna.split('.')[0]
        latent_fpa = os.path.join(img_dir, fna.replace('.jpg', '.npy'))
        img_fpa = os.path.join(img_dir, fna)
        hl_fpa1, hl_fpa2 = os.path.join(hl_mask_dir, fna_key + '.png'), os.path.join(hl_mask_dir, fna_key + '_hl.png')
        hl_mask_fpa = hl_fpa1 if os.path.exists(hl_fpa1) else hl_fpa2

        if not os.path.exists(img_fpa) or not os.path.exists(hl_mask_fpa):
            print(fna, ' not all files are found!')
            print(os.path.exists(img_fpa), img_fpa)
            print(os.path.exists(hl_mask_fpa), hl_mask_fpa)
            print('Quit!!!')
            quit()
            continue

        t_tar_img = trans_func(Image.open(img_fpa).convert("RGB"))
        hl_mask = read_mask(hl_mask_fpa, tar_size=256 if RETURN_EARLY is None else int(2**(RETURN_EARLY/2+2)))
        t_latent_d = np.load(latent_fpa, allow_pickle=True).item()
        cont_l.append((t_tar_img, hl_mask, None, t_latent_d, fna.split('.')[0]))


    return cont_l


def save_imgs_and_latents(args, round_idx, fpa_l, imgs, sample_w_l=None, sample_after_z=None):
    for idx, fpa in enumerate(fpa_l):
        fna = os.path.basename(fpa)
        new_fpa = os.path.join(args.img_out_dir, f'{fna}_pic-{int(np.random.rand()*10000)}.jpg')
        while os.path.exists(new_fpa):
            new_fpa = os.path.join(args.img_out_dir, f'{fna}_pic-{int(np.random.rand()*10000)}.jpg')


        Image.fromarray(imgs[idx]).save(new_fpa)
        if sample_w_l is not None:
            np.save(os.path.join(args.img_out_dir, new_fpa.replace('.jpg', '.npy')), {'w':[ sample_w_l[i][idx:idx+1,:,:]
                                                            for i in range(len(sample_w_l))], 'z':sample_after_z[idx]})


def init_generator(args, Generator, DEVICE='cuda'):
    style_dim = 512 if 'latent_dim' not in args else args.latent_dim
    n_mlp = 8 if 'n_mlp' not in args else args.n_mlp

    g_ema = Generator(args.img_size, style_dim, n_mlp)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(DEVICE)
    return g_ema


def read_batch_cont(batch_l, DEVICE='cuda'):
    imgs_l = [row[0] for row in batch_l]
    hl_mask_l = [row[1] for row in batch_l]
    grill_mask_l = [row[2] for row in batch_l]
    sampled_w_l = [torch.cat(row[3]['w'],1) for row in batch_l]
    raw_sampled_z = [row[3]['z'].unsqueeze(0) for row in batch_l]


    na_key_l = [row[4] for row in batch_l]

    imgs_npa = torch.stack(imgs_l, 0).to(DEVICE)
    hl_mask_npa = torch.stack(hl_mask_l, 0).to(DEVICE)
    sampled_w = torch.cat(sampled_w_l, 0).detach().to(DEVICE)
    sampled_z = torch.tensor(torch.cat(raw_sampled_z, 0).detach(), requires_grad=True)
    sampled_w.requires_grad = True
    hl_mask_npa.requires_grad = False

    return imgs_npa, hl_mask_npa, na_key_l, sampled_w, sampled_z



def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


def get_transform_fun(tar_size):

    transform = transforms.Compose([transforms.Resize(tar_size),
            transforms.CenterCrop(tar_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
    return transform


def read_imgs(tar_size, tar_pic_dir, out_dir, batch_size=16):
    transform = get_transform_fun(tar_size)
    img_whole_sets = []
    list_file_l = []

    candidate_s = set(os.listdir(tar_pic_dir))
    alread_done_s = set([fna.replace('-project.png', '.jpg') for fna in os.listdir(out_dir)])
    print('Already done size', len(alread_done_s))
    tar_l = list(candidate_s-alread_done_s)
    for idx in range(0,len(tar_l), batch_size):
        t_file_l, t_img_l = [], []
        for fna in tar_l[idx:idx+batch_size]:
            img_fpa = os.path.join(tar_pic_dir, fna)
            img = transform(Image.open(img_fpa).convert("RGB"))
            t_file_l.append(img_fpa)
            t_img_l.append(img)
        list_file_l.append(t_file_l)
        img_whole_sets.append(t_img_l)

    return list_file_l, img_whole_sets


def load_tar_dir_imgs(t_dir, min_thre, max_thre):
    t_transform_func = get_transform_fun(256)
    cand_l = os.listdir(t_dir)
    if min_thre is not None:
        cand_l = [ele for ele in cand_l if int(ele.split('.')[0]) >= min_thre and int(ele.split('.')[0]) < max_thre]

    batch_size = 16
    for i in range(0, len(cand_l), batch_size):
        batch = cand_l[i:i + batch_size]
        batch = [t_transform_func(Image.open(os.path.join(t_dir, fna)).convert("RGB")) for fna in batch]
        yield torch.stack(batch, 0)


@torch.no_grad()
def extract_feature_from_samples(
    g, inception, batch_size, n_sample, rand_latent_dim, device='cuda'):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, rand_latent_dim, device=device)
        img, _ = g([latent], truncation=1, truncation_latent=None)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))
    features = torch.cat(features, 0)
    return features


def get_generator_fid(args, g, train_test_benchmark,rand_latent_dim, device='cuda'):
    real_train_mean, real_train_cov, real_test_mean, real_test_cov = train_test_benchmark

    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()
    features = extract_feature_from_samples(g, inception, args.batch, args.fid_n_sample, rand_latent_dim,
                                            device).numpy()

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    fid_train = cal_fid(sample_mean, sample_cov, real_train_mean, real_train_cov)
    fid_test = cal_fid(sample_mean, sample_cov, real_test_mean, real_test_cov)
    return fid_train, fid_test


def get_dir_mean_cov_for_fid(t_dir, out_fpa, direct_return=False, min_thre=None, max_thre=None):
    from torch import nn

    inception = nn.DataParallel(load_patched_inception_v3()).to('cuda')
    # inception = load_patched_inception_v3().to('cuda')
    inception.eval()

    data_loader = load_tar_dir_imgs(t_dir, min_thre, max_thre)
    features = []
    for batch in tqdm(data_loader):
        feat = inception(batch)[0].view(batch.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0).numpy()

    if direct_return:
        return np.mean(features, 0), np.cov(features, rowvar=False)
    else:
        np.save(out_fpa, {'mean':np.mean(features, 0), 'cov':np.cov(features, rowvar=False)})


def cal_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace

    return fid


def get_FID(benchmark_save_fpa, tar_dir):
    benchmark_save = np.load(benchmark_save_fpa, allow_pickle=True).item()
    sample_mean, sample_cov = get_dir_mean_cov_for_fid(tar_dir, None, direct_return=True)
    fid_test = cal_fid(benchmark_save['mean'], benchmark_save['cov'], sample_mean, sample_cov)

    return fid_test


def my_percep_loss_FM( generated_FM, target, hl_mask):
    hl_mask = torch.tensor(hl_mask, device='cuda')
    hl_mask = hl_mask.unsqueeze(1).repeat(1, generated_FM.shape[1], 1, 1)
    mse = torch.mean(torch.abs((generated_FM * hl_mask) - (target.clone() * hl_mask)))

    return None, mse


def my_mse_loss(input, target, hl_mask):
    from torch.nn import _reduction as _Reduction

    if target.requires_grad:
        ret = (input - target) ** 2
        ret = torch.mean(ret)
    else:
        input = (input * torch.tensor(hl_mask, device='cuda'))
        target = (target * torch.tensor(hl_mask, device='cuda'))

        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum('mean'))
    return ret


def my_percep_loss(generated_imgs, target, hl_mask, PERCEPT):
    hl_mask = torch.tensor(hl_mask, device='cuda')
    hl_mask = hl_mask.unsqueeze(1).repeat(1, 3, 1, 1)

    generated_imgs = (generated_imgs * hl_mask)
    target = (target * hl_mask)
    lpips_loss = PERCEPT(generated_imgs, target).sum()
    mse = torch.mean((generated_imgs - target) ** 2)

    return lpips_loss, mse

def check_rst_img(t_latent, g_ema):
    img_gen, _ = g_ema(t_latent, input_is_latent=True, t_latent_n=14)
    Image.fromarray(make_image(img_gen)[0]).show()

def read_presaved_fid(t_fpa):
    t_save = np.load(t_fpa, allow_pickle=True).item()
    return t_save['mean'], t_save['cov']


def get_ppl_dist(args, g, percept, num_of_batches, device='cuda'):

    distances = []

    with torch.no_grad():
        for batch in tqdm(num_of_batches):
            noise = g.make_noise()

            inputs = torch.randn([batch * 2, args.style_dim], device=device)
            lerp_t = torch.rand(batch, device=device) if args.sampling == 'full' else torch.zeros(batch, device=device)

            if args.space == 'w':
                latent = g.get_latent(inputs)
                latent_t0, latent_t1 = latent[::2], latent[1::2]
                latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
                latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + args.eps)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

            image, _ = g([latent_e], input_is_latent=True, noise=noise)

            if args.crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode='bilinear', align_corners=False
                )

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (args.eps ** 2)
            distances.append(dist.to('cpu').numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_dist = np.extract(np.logical_and(lo <= distances, distances <= hi), distances)
    return filtered_dist