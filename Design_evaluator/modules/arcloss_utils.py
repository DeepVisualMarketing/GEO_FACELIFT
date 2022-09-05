# @Author : JM_Huang
# @Time   : 25/09/21

import os
import sys
from own_pathes import own_path_d

from Design_evaluator.modules.vgg16_torch import VGG16
from Design_evaluator.models.metrics import ArcMarginProduct
from Design_evaluator.models.focal_loss import FocalLoss
import torch.nn as nn
import torch


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


def load_pre_save_eval(vgg_model, class_head, reg_head, save_dir, epoch_num):
    vgg_fna = os.path.join(save_dir, f'vgg16_{epoch_num}.pth')
    cla_head_fna = os.path.join(save_dir, f'arc_head_{epoch_num}.pth')
    reg_head_fna = os.path.join(save_dir, f'reg_head_{epoch_num}.pth')

    vgg_model.load_state_dict(torch.load(vgg_fna))
    class_head.load_state_dict(torch.load(cla_head_fna))
    reg_head.load_state_dict(torch.load(reg_head_fna))

    vgg_model.defrozen_all_para()
    loaded_epoch = int(vgg_fna.split('.')[0].split('_')[-1])
    print('load presaved epoch', loaded_epoch)
    return vgg_model, class_head, reg_head