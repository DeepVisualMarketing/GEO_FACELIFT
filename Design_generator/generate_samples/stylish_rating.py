import os
import sys
from own_pathes import own_path_d
sys.path.append(own_path_d['TF'])
from CNNs.VGG16_latest import VGG16
from read_ini import read_ini_as_d
from tf_utils import count_rec_num
import tensorflow as tf
import numpy as np


# ini_pa = '/home/tech/Workspace/Projects/Facelift/stage_III/stylishness/train_setting2.ini'
#
# ini_d = read_ini_as_d(ini_pa)
# ini_d['paths']['data_file_1'] = '/home/tech/Workspace/Data/Projects_working/Facelift/Stage_III_evaluation/stylishness/stylish_rec_test.tfrecords'
# ini_d['paras']['dropout'] = False
# ini_d['paras']['no_shuff'] = True
# ini_d['paras']['data_size'] = 561
# ini_d['paras']['batch_size'] = 1
# ini_d['basic']['take_feed'] = 'True'
#
# tar_dir = '/home/tech/Workspace/Data/Project_tmp/Facelift/Stage_III_eval'
#
# for model_na in sorted(os.listdir(tar_dir), key= lambda x: int(x.split('_')[0])):
#     if int(model_na.split('_')[0])!=400:
#         continue
#     with tf.Graph().as_default():
#         model_fpa = os.path.join(tar_dir, model_na)
#         t_model = VGG16(ini_d)
#         tar_dir = '/home/tech/Workspace/Data/Project_tmp/Facelift/stage_II_exp/stylegan_10_06/setting_5_to_end_for_high'
#         tar_pic_l = [os.path.join(tar_dir, fna) for fna in os.listdir(tar_dir)]
#         sum_loss, stat_d = t_model.pred_on_pic_l(model_fpa, tar_pic_l, 'stylish_scores_high')

rst = np.load('/home/tech/Workspace/Projects/Facelift/stage_II/stylegan2/generate_samples/stylish_scores_high.npy', allow_pickle=True)
rst_d = rst.item()
print(np.sum([ele[0][0] for ele in list(rst_d.values())]))
