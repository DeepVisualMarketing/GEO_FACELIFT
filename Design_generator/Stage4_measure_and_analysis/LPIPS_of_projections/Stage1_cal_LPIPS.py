import os
import torch
import numpy as np
from PIL import Image
import Design_generator.lpips as lpips
from Design_generator.my_utils import get_transform_fun


source_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/projection_rst/raw'
DEVICE = 'cuda'
transform = get_transform_fun(256)
percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=DEVICE.startswith("cuda"))

if __name__ == '__main__':
    rst_d = {}
    for t_latent_n in [1,2,7,14]:
        projected_dir = f'/home/tech/Workspace/Data/Projects_working/Facelift/projection_rst/projected_uncong_imgs_{t_latent_n}'
        rst_d[t_latent_n] = {}
        tar_l = os.listdir(projected_dir)
        batch_size = 16
        for idx in range(0, len(tar_l), batch_size):
            print(idx)
            proj_img_l = []
            source_img_l = []
            img_na_l = []
            for fna in tar_l[idx:idx + batch_size]:
                img_na_l.append(fna)
                proj_fpa = os.path.join(projected_dir, fna)
                source_fpa = os.path.join(source_dir, fna.replace('-project.png', '.jpg'))
                proj_img = transform(Image.open(proj_fpa).convert("RGB"))
                source_img = transform(Image.open(source_fpa).convert("RGB"))
                proj_img_l.append(proj_img)
                source_img_l.append(source_img)

            proj_imgs = torch.stack(proj_img_l, 0).to(DEVICE)
            source_imgs = torch.stack(source_img_l, 0).to(DEVICE)
            p_loss = percept(proj_imgs, source_imgs).reshape((-1,1))
            for idx, fna in enumerate(img_na_l):
                rst_d[t_latent_n][fna.split('.')[0]] = p_loss[idx].detach().cpu().numpy()[0]

        # np.save('ws_dstore/LPIPS_of_different_latent_projections', rst_d)


