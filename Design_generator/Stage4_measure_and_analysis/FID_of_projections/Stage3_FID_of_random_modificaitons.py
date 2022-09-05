import os
import numpy as np
from scipy import linalg


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
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


if __name__ == '__main__':

    benchmark_save = np.load(
        '/home/tech/Workspace/Projects/Facelift/Stylegan2/Stage4_measure_and_analysis/checkpoints_FID_PPL/fid/ws_dstore/test_sample.npy', allow_pickle=True).item()
    npa_dir = '/home/tech/Workspace/Projects/Facelift/Stylegan2/Stage4_measure_and_analysis/FID_of_projections/ws_dstore/pure_projections'
    out_rec_fpa = '/home/tech/Workspace/Lib/py3_lib/Other_ppl_based_models/Stylegan2/measure_and_analysis/FID_of_projections/ws_dstore/random_proj_fid.csv'

    for t_latent_n in [1, 2, 7, 14]:
        t_proj = np.load(os.path.join(npa_dir, f'saved_mean_cov_{t_latent_n}.npy'), allow_pickle=True).item()

        fid_test = calc_fid(benchmark_save['mean'], benchmark_save['cov'], t_proj['mean'], t_proj['cov'])
        print(f'{t_latent_n},{fid_test}')
        # with open(out_rec_fpa, 'a') as f_out:
        #     f_out.write(f'{t_latent_n},{fid_test}\n')

