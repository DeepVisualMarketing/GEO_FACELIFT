import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import imread
import numpy as np
import tensorflow as tf
from torchvision import transforms
import os
from torch.hub import load_state_dict_from_url
from PIL import Image
from own_pathes import own_path_d
sys.path.append(own_path_d['Pic'])
from pic_utils import my_resize_img
from read_ini import read_ini_as_d


class VGG16(nn.Module):
    def __init__(self, cf_d):
        super(VGG16, self).__init__()
        self.cf_d = cf_d
        self.transform = self.get_transform_fun()
        self.drop_layer = nn.Dropout(p=0.5)
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

        # fully conected layers:
        self.fc6 = nn.Linear(7*7*512, self.cf_d['paras']['num_classes'])
        self.active_func = nn.LeakyReLU(0.1)
        # self.active_func = nn.ReLU()

    def forward(self, x):
        
        x = self.active_func(self.conv1_1(x))
        x = self.active_func(self.conv1_2(x))
        x = self.pool(x)

        x = self.active_func(self.conv2_1(x))
        x = self.active_func(self.conv2_2(x))
        x = self.pool(x)

        x = self.active_func(self.conv3_1(x))
        x = self.active_func(self.conv3_2(x))
        x = self.active_func(self.conv3_3(x))
        x = self.pool(x)

        x = self.active_func(self.conv4_1(x))
        x = self.active_func(self.conv4_2(x))
        x = self.active_func(self.conv4_3(x))
        x = self.pool(x)

        x = self.active_func(self.conv5_1(x))
        x = self.active_func(self.conv5_2(x))
        x = self.active_func(self.conv5_3(x))
        x = self.pool(x)

        x = x.view(-1, 7 * 7 * 512)
        x = self.fc6(x)
        # x = self.drop_layer(x)
        return x

    @staticmethod
    def get_transform_fun():
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'eval': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return data_transforms

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = self.forward(x)
        print('before pred', torch.max(x, 1))
        x = F.softmax(x)
        return x

    def accuracy(self, x, y):
        prediction = self.predict(x)
        maxs, indices = torch.max(prediction, 1)
        acc = 100 * torch.sum(torch.eq(indices.float(), y.float()).float())/y.size()[0]
        return acc.cpu().data[0]

    def load_pretrained_weights(self):
        source_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth', progress=True)

        cur_state_d = self.state_dict()

        if self.cf_d['basic']['load_type'] not in ['no_head', 'full']:
            print('Unrecognised load type', self.cf_d['basic']['load_type'])
            quit()
        else:
            for (tar_key_na, source_key_na) in zip(list(cur_state_d.keys()), list(source_state_dict.keys())):
                print('load matching', tar_key_na, source_key_na)
                if self.cf_d['basic']['load_type'] == 'no_head':
                    if tar_key_na == 'fc6.weight':
                        nn.init.xavier_uniform(cur_state_d[tar_key_na])
                    elif tar_key_na == 'fc6.bias':
                        nn.init.constant(cur_state_d[tar_key_na], 0)
                    else:
                        cur_state_d[tar_key_na].copy_(source_state_dict[source_key_na])
                else:
                    cur_state_d[tar_key_na].copy_(source_state_dict[source_key_na])

        # -- Decide the frozen parameters
        if self.cf_d['basic']['frozen_type'] not in ['only_free_fcs', 'only_free_fc8', 'no_lock']:
            print('Unrecognised lock type', self.cf_d['basic']['frozen_type'])
        else:
            for name, para in self.named_parameters():
                if self.cf_d['basic']['frozen_type'] == 'only_free_fcs' and 'fc' not in name:
                    para.requires_grad = False
                    print(name, 'locked')
                elif self.cf_d['basic']['frozen_type'] == 'only_free_fc8' and 'fc8' not in name:
                    para.requires_grad = False
                    print(name, 'locked')
                elif self.cf_d['basic']['frozen_type'] == 'all_free':
                    continue

    def defrozen_all_para(self):
        for name, para in self.named_parameters():
            para.requires_grad = True


def train_one_epoch(t_model):
    optimizer = torch.optim.SGD(t_model.parameters(), lr=0.0002)
    criterion = torch.nn.MSELoss()

    img_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/data/big_one/raw'

    for img_fna in os.listdir(img_dir):
        image = vgg16.transform['eval'](Image.open(os.path.join(img_dir, img_fna)))
        batch_t = torch.unsqueeze(image, 0)
        y = t_model(batch_t.cuda())
        y_hat = int(img_fna.split('$$')[2]) - 2000
        # print(y, y_hat)
        loss = criterion(y, torch.tensor(y_hat).float().cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    cf_d = read_ini_as_d('/home/tech/Workspace/Projects/Small_projects/Learning/ws_dstore/t_setting.ini')
    vgg16 = VGG16(cf_d)

    imagenet_trained_vgg16_fpa = '/home/tech/Workspace/Data/Classical_dl_model_paras/VGG16/vgg_16.ckpt'

    vgg16.load_pretrained_weights()
    vgg16.cuda()
    train_one_epoch(vgg16)


    # img_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/data/big_one/raw'
    #
    # for img_fna in os.listdir(img_dir):
    #     image = vgg16.transform['eval'](Image.open(os.path.join(img_dir, img_fna)))
    #     batch_t = torch.unsqueeze(image, 0)
    #     print(torch.max(vgg16.predict(batch_t.cuda()), 1))


