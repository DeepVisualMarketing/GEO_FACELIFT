# @Author : JM_Huang
# @Time   : 31/08/20

import os
import sys
from own_pathes import own_path_d
import argparse

def read_setting():
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="/home/tech/Workspace/Data/Project_tmp/Facelift/style_GAN/10-06/checkpoint/250000.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--saved_latent_pa', type=str, default='/home/tech/Workspace/Projects/Facelift/stage_II/stylegan2/projector/low_stylish_insp_0.pt')
    return parser.parse_args()