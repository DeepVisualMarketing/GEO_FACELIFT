import os
import numpy as np
from PIL import Image
from Design_generator.tools.pic_utils import enlarge_mask


def get_the_box(t_mask):
    xs_l = [idx for idx, val in enumerate(list(np.max(t_mask, axis=0))) if val == True]
    ys_l = [idx for idx, val in enumerate(list(np.max(t_mask, axis=1))) if val == True]
    return ys_l[0], ys_l[-1], xs_l[0], xs_l[-1]


def get_mask_position(t_mask):
    mid_idx = int(t_mask.shape[1]/2)
    left_mask, right_mask = np.copy(t_mask), np.copy(t_mask)
    left_mask[:, mid_idx:] = False
    right_mask[:, :mid_idx] = False
    left_hl_pos = get_the_box(left_mask)
    right_hl_pos = get_the_box(right_mask)

    return left_hl_pos, right_hl_pos


def read_mask(t_fpa, expend_size=25):
    t_mask = np.array(Image.open(t_fpa).resize((300,300)))
    t_mask = t_mask[:, :, -1] > 200

    t_mask = enlarge_mask(t_mask, expend_size=expend_size)
    left_hl_pos, right_hl_pos = get_mask_position(t_mask)
    return t_mask, left_hl_pos, right_hl_pos


def get_all_masks(tar_dir):
    rst_d = {}
    for fna in os.listdir(tar_dir):
        t_mask, left_hl_pos, right_hl_pos = read_mask(os.path.join(tar_dir, fna))
        rst_d[fna.split('.')[0]] = (t_mask, left_hl_pos, right_hl_pos)
    return rst_d


def extract_tar_pos(img_npa, left_hl_pos, right_hl_pos):
    y_min, y_max, x_min, x_max = right_hl_pos
    right_hl = img_npa[y_min:y_max,x_min:x_max,:]
    y_min, y_max, x_min, x_max = left_hl_pos
    left_hl = img_npa[y_min:y_max,x_min:x_max,:]

    return left_hl, right_hl


def zoom_given_img_hl(img_fpa, mask_fpa, expend_size=25, adjust=None):

    t_img_npa = np.array(Image.open(img_fpa).resize((300,300)))

    t_mask, left_hl_pos, right_hl_pos = read_mask(mask_fpa, expend_size=expend_size)
    if adjust is not None:
        left_hl_pos = list(left_hl_pos)
        left_hl_pos[adjust[0]] += adjust[1]
    left_hl_npa, right_hl_npa = extract_tar_pos(t_img_npa, left_hl_pos, right_hl_pos)

    zoom_left_npa = np.array(Image.fromarray(left_hl_npa).resize((left_hl_npa.shape[1]*2, left_hl_npa.shape[0]*2)))
    t_img_npa[:zoom_left_npa.shape[0], :zoom_left_npa.shape[1],:] = zoom_left_npa

    return Image.fromarray(t_img_npa)


if __name__ == '__main__':
    tar_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/target_models/masks/hl_mask'
    out_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/target_models/upgraded_rst/tmp'
    design_dir = '/home/tech/Workspace/Data/Projects_working/Facelift/target_models/upgraded_rst/imgs_compare_k'
    mask_d = get_all_masks(tar_dir)

    for fna in os.listdir(design_dir):
        t_img = Image.open(os.path.join(design_dir, fna)).resize((300,300))
        t_img_npa = np.array(t_img)
        key = fna.split('_lr')[0]
        left_hl_npa, right_hl_npa = extract_tar_pos(t_img_npa, *mask_d[key][1:])

        zoom_left_npa = np.array(Image.fromarray(left_hl_npa).resize((left_hl_npa.shape[1]*2, left_hl_npa.shape[0]*2)))

        t_img_npa[:zoom_left_npa.shape[0], :zoom_left_npa.shape[1],:] = zoom_left_npa

        Image.fromarray(t_img_npa).save(os.path.join(out_dir, fna))

