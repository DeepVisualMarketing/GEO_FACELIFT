from __future__ import print_function
import os, sys
import torch
import time
import argparse
import numpy as np
import torch.nn as nn
from shutil import copyfile
import torch.nn.functional as F
from read_ini import read_ini_as_d
from torch.optim.lr_scheduler import StepLR
from Design_evaluator.modules.vgg16_torch import VGG16
from Design_evaluator.models.metrics import ArcMarginProduct
from Design_evaluator.modules.config import Config
import torchvision.datasets as datasets
import torch.utils.data as th_data
from Design_evaluator.modules.MyTorchDataset import MyTorchDataset
torch.manual_seed(17)


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def initil_models(args, class_len, device):

    vgg16 = VGG16(args.vgg_settign_d)
    vgg16.load_pretrained_weights()
    vgg16.to(device)
    vgg16.train()

    metric_fc = ArcMarginProduct(512, out_features=class_len, s=30, m=0.5, easy_margin=False)
    metric_fc.to(device)
    reg_head = nn.Linear(512, 1).to(device)

    class_criterion = torch.nn.CrossEntropyLoss()  # or criterion = FocalLoss(gamma=2)
    reg_criterion = torch.nn.MSELoss()
    return vgg16, metric_fc, reg_head, class_criterion, reg_criterion


def get_classification_dataloaders(args, batch_size):
    whole_set = datasets.ImageFolder(args.img_root_dir, transform=VGG16.get_transform_fun()['eval'])
    print('data size', len(whole_set))
    train_set, test_set = th_data.random_split(whole_set, [len(whole_set)-2000, 2000])
    train_data_generator = th_data.DataLoader(train_set, shuffle=True, batch_size=batch_size) #, sampler=torch.utils.data.RandomSampler()
    test_data_generator = th_data.DataLoader(train_set, batch_size=batch_size)
    return train_data_generator, test_data_generator, whole_set.class_to_idx


def get_regression_dataloaders(batch_size, tar_dir_l, rating_fpa, no_train=False, tar_set_idx=None):

    ratins_d = None if rating_fpa is None else np.load(rating_fpa, allow_pickle=True).item()
    my_dataset = MyTorchDataset(img_dir_l=tar_dir_l, label_d=ratins_d, transform=VGG16.get_transform_fun()['eval'])

    if no_train:
        train_set, test_set = th_data.random_split(my_dataset, [0, len(my_dataset)])
        train_data_generator = None
    else:
        train_set, test_set = th_data.random_split(my_dataset, [len(my_dataset) - 1000, 1000])
        train_data_generator = th_data.DataLoader(train_set, shuffle=True,
                                                  batch_size=batch_size)
    print('regression train size', len(train_set), 'regression test size', len(test_set))

    test_data_generator = th_data.DataLoader(test_set, batch_size=batch_size)
    return train_data_generator, test_data_generator, len(my_dataset.class_l)


def load_pre_save(vgg_model, head_model, save_dir):
    vgg_fna = sorted([fna for fna in os.listdir(save_dir) if 'vgg' in fna], key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    head_fna = sorted([fna for fna in os.listdir(save_dir) if 'head' in fna], key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    vgg_model.load_state_dict(torch.load(os.path.join(save_dir, vgg_fna)))
    head_model.load_state_dict(torch.load(os.path.join(save_dir, head_fna)))
    vgg_model.defrozen_all_para()
    loaded_epoch = int(vgg_fna.split('.')[0].split('_')[-1])
    print('load presaved epoch', loaded_epoch)
    return vgg_model, head_model, loaded_epoch


def assign_models():
    img_root_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/data/large_one/asethetic_finnal_data_filter_ratings_less_than_80'
    stat_d = {}
    store_l = [[idx, 0, []] for idx in range(5)]

    for fna in os.listdir(img_root_dir):
        model = '$$'.join(fna.split('$$')[:2])
        if model not in stat_d:
            stat_d[model] = 0
        stat_d[model] += 1


    for model, pic_num in sorted(stat_d.items(), key=lambda x: x[1], reverse=True):
        smallest_set = sorted(store_l, key=lambda x: x[1])[0][0]
        store_l[smallest_set][2].append(model)
        store_l[smallest_set][1]+= pic_num
    np.save('ws_dstore/assigned_models', store_l)


def create_split_set(img_root_dir, out_root_dir):
    rst_l = np.load('ws_dstore/assigned_models.npy', allow_pickle=True)
    for fna in os.listdir(img_root_dir):
        model = '$$'.join(fna.split('$$')[:2])
        tar_idx = [idx for idx, pic_num, model_l in rst_l if model in model_l][0]
        out_dir = os.path.join(out_root_dir, str(tar_idx))
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        copyfile(os.path.join(img_root_dir, fna), os.path.join(out_dir, fna))


def load_pre_save_eval(vgg_model, class_head, reg_head, save_dir, epoch_num):
    vgg_fna = os.path.join(save_dir, f'vgg16_{epoch_num}.pth')
    reg_head_fna = os.path.join(save_dir, f'reg_head_{epoch_num}.pth')

    vgg_model.load_state_dict(torch.load(vgg_fna))
    reg_head.load_state_dict(torch.load(reg_head_fna))

    vgg_model.defrozen_all_para()
    loaded_epoch = int(vgg_fna.split('.')[0].split('_')[-1])
    print('load presaved epoch', loaded_epoch)
    return vgg_model, class_head, reg_head


def get_para():
    args = argparse.ArgumentParser().parse_args()

    args.img_root_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/data/large_one/tmp' #aesthetic_n_folder
    args.rating_fpa = '/home/tech/Workspace/Projects/Facelift/evaluator_arcloss/my_own/ws_dstore/pic_ratings_d_filter_80.npy'
    args.train_batch_size = 16  # batch size
    args.test_batch_size = 60
    args.lr = 1e-4
    args.print_freq = 1000
    args.max_epoch = 60
    args.save_interval = 2
    args.weight_decay = 5e-6
    args.checkpoints_dir = '/home/tech/Workspace/Data/Project_tmp/Facelift/evaluator_arcface/rst/210416'
    args.vgg_settign_d = read_ini_as_d('/home/tech/Workspace/Projects/Facelift/evaluator_arcloss/my_own/t_setting.ini')

    return torch.device("cuda"), args


if __name__ == '__main__':
    device, args = get_para()

    rst_d = {}
    with torch.no_grad():
        for tar_set_idx in range(5):
            rst_d[tar_set_idx] = {}
            tar_dir_l = [os.path.join(args.img_root_dir, str(i)) for i in range(5) if i != tar_set_idx]
            reg_train_data_generator, reg_test_data_generator, class_len = get_regression_dataloaders(args.train_batch_size,
                                                                    tar_dir_l, args.rating_fpa, tar_set_idx=tar_set_idx)
            _, reg_valid_data_generator, _ = get_regression_dataloaders(args.train_batch_size,
                [os.path.join(args.img_root_dir, str(tar_set_idx))], args.rating_fpa, no_train=True, tar_set_idx=tar_set_idx)
            # -- Settings
            checkpoints_path = os.path.join(args.checkpoints_dir, str(tar_set_idx))
            if not os.path.exists(checkpoints_path): os.makedirs(checkpoints_path)

            loaded_epoch = -1
            vgg_model, class_head, reg_head, class_criterion, reg_criterion = initil_models(args, class_len, device)
            optimizer = torch.optim.Adam([{'params': vgg_model.parameters()}, {'params': reg_head.parameters()}],  #{'params': class_head.parameters()}
                                         lr=args.lr, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=6, gamma=0.92)
            start = time.time()

            sum_loss_reg = 0
            for i in range(0, 60, 2):
                print('epoch', i)
                sum_loss_classs, sum_loss_reg, sum_acc = 0, 0, 0
                vgg_model, class_head, reg_head = load_pre_save_eval(vgg_model, class_head, reg_head,
                                                                              checkpoints_path, epoch_num=i)
                vgg_model.eval()
                rst_d[tar_set_idx][i] = {'test_reg_real': [], 'test_reg_pred': [], 'valid_reg_real':[], 'valid_reg_pred':[]}

                for ii, data in enumerate(reg_valid_data_generator):
                    reg_data, labels = data
                    img_na, reg_label, class_label = labels
                    reg_label = reg_label.to(device).float()

                    feature = vgg_model(reg_data.to(device))

                    reg_output = reg_head(feature)
                    reg_loss = reg_criterion(reg_output, reg_label.unsqueeze(1))

                    sum_loss_reg += reg_loss.cpu().detach().numpy()
                    rst_d[tar_set_idx][i]['valid_reg_real'] += list(reg_label.cpu().detach().numpy())
                    rst_d[tar_set_idx][i]['valid_reg_pred'] += list(reg_output.cpu().detach().numpy())

                rst_d[tar_set_idx][i]['valid_reg_los'] = sum_loss_reg
                print(tar_set_idx, i, sum_loss_reg)
                sum_loss_classs, sum_loss_reg, sum_acc = 0, 0, 0

                # output_na = 'no_classification_test_rst' if '210910_no_classificaiton' in args.checkpoints_dir else 'arc_reg_loss'
                # np.save(output_na, rst_d)
                # np.save('new_with_class', rst_d)

