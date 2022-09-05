import os
import torch
import argparse
from Design_generator.lpips import PerceptualLoss
from Design_generator.tools.original_model import Generator
from Design_generator.my_utils import init_generator, read_presaved_fid, cal_fid, get_generator_fid, get_ppl_dist
torch.manual_seed(17)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--fid_n_sample', type=int, default=10001)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--inception', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default='')
    # the folloing is for PPL
    parser.add_argument('--space', default='w', choices=['z', 'w'])
    parser.add_argument('--ppl_batch', type=int, default=16)
    parser.add_argument('--ppl_n_sample', type=int, default=5000)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--sampling', default='end', choices=['end', 'full'])

    args = parser.parse_args()
    args.style_dim=2048
    args.n_mlp=3
    args.fid_bench_fpa = '/home/tech/Workspace/Projects/Facelift/Stylegan2/Stage4_measure_and_analysis/checkpoints_FID_PPL/fid/ws_dstore'
    args.ckpt_dir = '/home/tech/Workspace/Data/Project_tmp/Facelift/style_GAN/210826/checkpoint'

    return args

if __name__ == '__main__':
    args = get_args()

    real_test_mean, real_test_cov = read_presaved_fid(os.path.join(args.fid_bench_fpa, 'test_sample.npy'))
    real_train_mean, real_train_cov = read_presaved_fid(os.path.join(args.fid_bench_fpa, 'train_sample.npy'))

    percept = PerceptualLoss(model='net-lin', net='vgg', use_gpu='cuda'.startswith('cuda'))
    ppl_batch_sizes = [args.ppl_batch] * (args.ppl_n_sample // args.ppl_batch) + [args.ppl_n_sample % args.ppl_batch]

    cand_l = [fna for fna in os.listdir(args.ckpt_dir) if int(fna.split('.')[0]) > 102500]
    for fna in sorted(cand_l, key=lambda x: int(x.split('.')[0])):
        print('start', fna)
        args.ckpt = os.path.join(args.ckpt_dir, fna)
        t_g = init_generator(args, Generator)

        fid_train, fid_test = get_generator_fid(args, t_g, [real_train_mean, real_train_cov, real_test_mean, real_test_cov], args.style_dim)

        with open('new_210826_fid.csv', 'a') as f_out:
            f_out.write(f'{fna},{fid_train},{fid_test}\n')

        # filtered_dist = get_ppl_dist(args, t_g, percept, ppl_batch_sizes, device='cuda')
        # with open('ppl_of_checkpoints.csv', 'a') as f_out:
        #     f_out.write(f'{fna},{filtered_dist.mean()}\n')

