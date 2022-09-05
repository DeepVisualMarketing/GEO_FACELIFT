from PIL import Image
import os
from Design_evaluator.modules.vgg16_torch import VGG16
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyTorchDataset(Dataset):
    def __init__(self, img_dir_l, label_d, transform=None):
        self.img_dir_l = img_dir_l
        self.img_fpa_l = []
        for img_dir in img_dir_l:
            for fna in os.listdir(img_dir):
                self.img_fpa_l.append(os.path.join(img_dir, fna))

        self.transform = transform
        self.label_d = label_d
        self.class_l = sorted(list(set(tuple(fpa.split('/')[-1].split('$$')[:2]) for fpa in self.img_fpa_l)))

    def get_image_from_folder(self, fpa):
        image = Image.open(fpa)
        return image

    def __len__(self):
        return len(self.img_fpa_l)

    def __getitem__(self, index):
        X = self.get_image_from_folder(self.img_fpa_l[index])
        fna = self.img_fpa_l[index].split('/')[-1]
        model_key = tuple(fna.split('$$')[:2])

        Y0 = fna
        Y1 = False if self.label_d is None else self.label_d[fna]*10
        Y2 = False if model_key not in self.class_l else self.class_l.index(model_key)

        if self.transform is not None:
            X = self.transform(X)

        return [X, (Y0, Y1, Y2)]

if __name__ == '__main__':
    ratins_d = np.load('/home/tech/Workspace/Projects/Facelift/evaluator_arcloss/my_own/ws_dstore/pic_ratings_d.npy', allow_pickle=True).item()

    my_dataset = MyTorchDataset(img_dir='/home/tech/Workspace/Data/Projects_working/Facelift/data/large_one/for_aesthetic_ratings', label_d=ratins_d, transform=VGG16.get_transform_fun()['train'])
    data_loader = DataLoader(my_dataset, shuffle=True, batch_size=4)  # , sampler=torch.utils.data.RandomSampler()
    t_iter = data_loader.__iter__()

    num = 0
    for i in range(20000):
        num += 1
        if num%100 ==0:
            print('process', num)
        t_iter.next()
    print(num)