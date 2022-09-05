from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyPICDataset(Dataset):
    def __init__(self, img_dir_l, exist_l=None, transform=None, tar_type=None):
        self.img_dir_l = img_dir_l
        self.img_fpa_l = []
        for img_dir in img_dir_l:
            for fna in os.listdir(img_dir):
                if tar_type is not None and tar_type not in fna: continue

                if exist_l is not None and fna in exist_l: continue
                self.img_fpa_l.append(os.path.join(img_dir, fna))

        self.transform = transform

    def __len__(self):
        return len(self.img_fpa_l)

    def __getitem__(self, index):
        img_fpa = self.img_fpa_l[index]
        X = Image.open(img_fpa)
        fna = img_fpa.split('/')[-1]

        if self.transform is not None:
            X = self.transform(X)

        return np.array(X), img_fpa

if __name__ == '__main__':
    None