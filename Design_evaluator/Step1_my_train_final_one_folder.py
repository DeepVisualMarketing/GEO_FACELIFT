from __future__ import print_function
import os, sys
import torch
import time
import argparse
import numpy as np
from shutil import copyfile
from Design_evaluator.modules.vgg16_torch import VGG16
import torch.utils.data as th_data
from read_ini import read_ini_as_d
import torchvision.datasets as datasets
from Design_evaluator.modules.arcloss_utils import initil_models
from Design_evaluator.modules.MyTorchDataset import MyTorchDataset
from torch.optim.lr_scheduler import StepLR
torch.manual_seed(17)


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def get_classification_dataloaders(args, batch_size):
    whole_set = datasets.ImageFolder(args.img_root_dir, transform=VGG16.get_transform_fun()['train'])
    print('data size', len(whole_set))
    train_set, test_set = th_data.random_split(whole_set, [len(whole_set)-2000, 2000])
    train_data_generator = th_data.DataLoader(train_set, shuffle=True, batch_size=batch_size) #, sampler=torch.utils.data.RandomSampler()
    test_data_generator = th_data.DataLoader(train_set, batch_size=batch_size)
    return train_data_generator, test_data_generator, whole_set.class_to_idx


def get_regression_dataloaders(batch_size, tar_dir_l, rating_fpa, no_train=False, tar_set_idx=None):

    ratins_d = None if rating_fpa is None else np.load(rating_fpa, allow_pickle=True).item()
    my_dataset = MyTorchDataset(img_dir_l=tar_dir_l, label_d=ratins_d, transform=VGG16.get_transform_fun()['train'])

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


def assign_models(img_root_dir):
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


def get_para():
    args = argparse.ArgumentParser().parse_args()

    args.img_root_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/data/large_one/aesthetic_n_folder'
    args.rating_fpa = '/home/tech/Workspace/Projects/Facelift/evaluator_arcloss/my_own/ws_dstore/pic_ratings_d_filter_80.npy'
    args.train_batch_size = 16  # batch size
    args.test_batch_size = 60
    args.lr = 1e-4
    args.print_freq = 1000
    args.max_epoch = 100
    args.save_interval = 2
    args.weight_decay = 5e-6
    args.checkpoints_dir = '/home/tech/Workspace/Data/Project_tmp/Facelift/evaluator_arcface/rst/211114_final_one'
    args.vgg_settign_d = read_ini_as_d('/home/tech/Workspace/Projects/Facelift/evaluator_arcloss/my_own/ws_dstore/settings/vgg_setting.ini')
    args.loaded_epoch = -1

    return torch.device("cuda"), args


if __name__ == '__main__':
    device, args = get_para()
    log_l = []

    tar_dir_l = [os.path.join(args.img_root_dir, str(i)) for i in range(5)]
    reg_train_data_generator, reg_test_data_generator, class_len = get_regression_dataloaders(args.train_batch_size,
                                                            tar_dir_l, args.rating_fpa)
    # -- Settings
    checkpoints_path = os.path.join(args.checkpoints_dir)
    if not os.path.exists(checkpoints_path): os.makedirs(checkpoints_path)

    vgg_model, class_head, reg_head, class_criterion, reg_criterion = initil_models(args, class_len, device)
    optimizer = torch.optim.Adam([{'params': vgg_model.parameters()}, {'params': reg_head.parameters()}],  #{'params': class_head.parameters()}
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.92)

    # -- Start training
    sum_loss_classs, sum_loss_reg,sum_acc = 0, 0, 0
    for i in range(args.loaded_epoch+1, args.max_epoch):  #!!! Make sure the class number is right!!!
        print('epoch', i, 'learning rate' , optimizer.param_groups[0]['lr'])

        if i == 11:
            vgg_model.defrozen_all_para()

        for ii, data in enumerate(reg_train_data_generator):
            reg_data, labels = data
            img_na, reg_label, class_label = labels
            class_label = class_label.to(device).long()
            reg_label = reg_label.to(device).float()
            feature = vgg_model(reg_data.to(device))

            class_output, class_real_pred = class_head(feature, class_label)
            class_loss = class_criterion(class_output, class_label)
            reg_output = reg_head(feature)
            reg_loss = reg_criterion(reg_output, reg_label.unsqueeze(1))
            total_loss = reg_loss + class_loss*(100-i)/100.0

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sum_loss_classs += class_loss.cpu().detach()
            sum_loss_reg += reg_loss.cpu().detach()
            sum_acc += np.sum((np.argmax(class_real_pred.data.cpu().numpy(), axis=1) == class_label.data.cpu().numpy()))

        time_str = time.asctime(time.localtime(time.time()))
        log_l.append([('dataset_group', 0), ('epoch', i), ('sum_loss_reg', sum_loss_reg),
                     ('sum_loss_classs', sum_loss_classs), ('sum_acc', sum_acc)] )

        if i % 1 == 0: np.save('no_classification_training_log', log_l)
        print(log_l[-1])
        sum_loss_classs, sum_loss_reg, sum_acc = 0, 0, 0
        start = time.time()

        if i % args.save_interval == 0 or i == args.max_epoch:
            save_model(vgg_model, checkpoints_path, 'vgg16', i)
            save_model(class_head, checkpoints_path, 'arc_head', i)
            save_model(reg_head, checkpoints_path, 'reg_head', i)
        scheduler.step()
        # np.save('training_log', log_l)

