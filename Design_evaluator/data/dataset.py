import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
from glob import glob


class Dataset(data.Dataset):

    def __init__(self, root, phase='train', input_shape=(3, 224, 224)):
        self.phase = phase
        self.input_shape = input_shape
        self.lable_d = {model: idx for idx, model in enumerate(os.listdir(root))}

        imgs = glob(os.path.join(root, '*/*.jpg'))
        self.imgs = np.random.permutation(imgs)

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        img_path = sample
        data = Image.open(img_path)
        # data = data.convert('L')
        data = self.transforms(data)
        label = self.lable_d[sample.split('/')[-2]]
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='/data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      phase='test',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)