import os, cv2
import numpy as np
from PIL import Image
from collections import Counter
from resizeimage import resizeimage
import scipy.ndimage.morphology as morphology


def conver_ps_png_to_jpg(t_fpa):
    '''
    This function process the photoshop processed images to jpg, which drop the alpha channel
    :param t_fpa:
    :return:
    '''
    img_npa = np.array(Image.open(t_fpa))
    t_mask = img_npa[:, :, -1] > 150
    for i in range(3):
        img_npa[:, :, i] = t_mask*img_npa[:, :, i] + (1-t_mask)*np.ones(img_npa.shape[:2])*255

    return img_npa[:,:,:3], t_mask


def label_individual_objs_of_binary_matrix(in_mask, get_boundary=False):
    label_image = label(in_mask)

    if get_boundary:
        boundary_l = []
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            boundary_l.append(region.bbox)
        return boundary_l
    else:
        return label_image


def check_start_end(in_line):
    start = None
    end = None
    in_l = list(in_line)
    t_length = len(in_l)
    for i in range(int(t_length/2)):
        if in_l[i] != 0:
            start = i
            break

    for i in range(int(t_length/2), t_length):
        if in_l[i] == 0:
            end = i
            break
    if start is None or end is None:
        return None, None
    tencent = (end - start)/20
    # return start, end, max(0, int(start - tencent)), min(t_length, int(end + tencent))
    return max(0, int(start - tencent)), min(t_length, int(end + tencent))


def crop_img_n_mask(mask_matx, rst_array):

    x_sum = np.sum(mask_matx, axis=0)
    y_sum = np.sum(mask_matx, axis=1)
    start_x, end_x = check_start_end(x_sum)
    start_y, end_y = check_start_end(y_sum)
    if None in [start_x, end_x, start_y, end_y]:
        return None,None

    start_y = max(0, start_y-7)
    end_y = min(mask_matx.shape[0], end_y + 7)

    new_img_array = rst_array[start_y:end_y, start_x:end_x,:]
    new_mask_matx = mask_matx[start_y:end_y, start_x:end_x]
    return new_img_array, new_mask_matx


def add_blue_highlight_mast(img_npa, guidbp_npa, out_fpa):
    """
    :param img_npa: Numpy array of image
    :param guidbp_npa: Numpy array of guided map
    :param out_fpa: the file path of output
    :return:
    """
    if 'White' not in out_fpa:
        return None
    img_npa = img_npa.astype(np.float32)
    img_npa[:,:,-1] += guidbp_npa
    img_npa[img_npa>255] = 255
    img_npa = img_npa.astype(np.uint8)
    Image.fromarray(img_npa).save(out_fpa)
    Image.fromarray(guidbp_npa).save(out_fpa.replace('.jpg', '.png'))


def whiten(img_npa, mask_npa):
    mask = enlarge_mask(mask_npa, expend_size=8)
    img_npa[mask == False] = (255, 255, 255)

    return img_npa


def whiten_and_resize(img_npa, mask_fpa):
    img = Image.fromarray(whiten(img_npa, mask_fpa))
    resized_img = my_resize_img(img, bg_color='white', tar_shape_l=[300, 300])
    return resized_img


def resize_mask(t_mask, tar_shape_l=[660, 270]):
    t_empty_rows = np.zeros((7,t_mask.shape[1]))
    new_mask = np.concatenate((t_empty_rows, t_mask, t_empty_rows), axis=0)
    resied_mask_img = my_resize_img(Image.fromarray((new_mask).astype('uint8')), bg_color='black',
                                    tar_shape_l=tar_shape_l)
    resied_mask = np.array(resied_mask_img)[:, :, 0] > 0
    return resied_mask

def resize_mask_2(mask_in, tar_size):
    resize_img = Image.fromarray(mask_in.astype(np.uint8) * 255).convert("L").resize((tar_size, tar_size), resample=Image.NEAREST)
    return np.array(resize_img) > 125


def my_resize_img(img, bg_color='white', tar_shape_l=[512, 512]):
    if bg_color =='white':
        bg_val = (255, 255, 255)
    else:
        bg_val = (0, 0, 0)
    resized_img = resizeimage.resize_contain(img, tar_shape_l, bg_color=bg_val).convert('RGB')
    return resized_img


def own_normalize(in_tensor):
    ten_intv = np.max(in_tensor) - np.min(in_tensor)
    new_tensor = (in_tensor - np.min(in_tensor)) / float(ten_intv)
    return new_tensor


def npa_to_image(in_tensor, scal_power=None):
    in_tensor = own_normalize(in_tensor)
    if scal_power is not None:
        in_tensor = np.power(in_tensor, scal_power)
        in_tensor = own_normalize(in_tensor)
    in_tensor = in_tensor * 255
    in_tensor = in_tensor.astype('uint8')

    return in_tensor


def get_pixel_freq(img_array):
    """
    To count the frequency of pixel values in the image
    :param img_array: The np-array of image
    :return: A dict
    """

    t_l = []
    for col in img_array:
        for pixel in col:
            t_l.append(tuple(pixel))

    return dict(Counter(t_l))


def reverse_my_vgg_resize(t_img_npa, ori_shape):
    x_larger_y = True if ori_shape[1] > ori_shape[0] else False
    ori_dim_diff = ori_shape[1] - ori_shape[0] if x_larger_y is True else ori_shape[0] - ori_shape[1]
    processed_dim_diff = int(float(ori_dim_diff)/ori_shape[int(x_larger_y)]*224/2)

    if x_larger_y is True:
        reshaped_img_npa = t_img_npa[processed_dim_diff:t_img_npa.shape[0] - processed_dim_diff, :]
    else:
        reshaped_img_npa = t_img_npa[:, processed_dim_diff:t_img_npa.shape[1] - processed_dim_diff]
    return reshaped_img_npa


def add_quadrangle_to_mask(xs, ys, given_mask):
    cand_img = np.zeros((*given_mask.shape[:2], 3))
    cand_img[:, :, 0 ] = given_mask

    rst_rec = np.concatenate((np.expand_dims(xs, 1), np.expand_dims(ys, 1)), axis=1)
    rst_rec = np.expand_dims(np.array(rst_rec), 0).astype('int')
    cand_img = cv2.fillPoly(cand_img, rst_rec, color=(1, 1, 1))
    return cand_img[:, :, 0]


def fix_wrong_expand(mask_ny, expand_size):
    expend_size_2 = int(expand_size/2)+1

    new_mask_ny = np.copy(mask_ny)
    for y in range(mask_ny.shape[0]):
        for x in range(mask_ny.shape[1]):

            if y < expend_size_2 or (mask_ny.shape[0]-y < expend_size_2) \
                    or x < expend_size_2 or (mask_ny.shape[1]-x < expend_size_2):
                for z in range(3):
                    new_mask_ny[y, x] = 1
    return new_mask_ny


def enlarge_mask(mask_ny, expend_size=15):
    mask_ny = np.invert(mask_ny.astype('bool'))
    mask_ny = morphology.binary_erosion(mask_ny, structure=np.ones((expend_size, expend_size)))
    mask_ny = fix_wrong_expand(mask_ny, expend_size)

    return np.invert(mask_ny.astype('bool'))


def cal_thred(img_npa, qual=4):
    new_l = []
    for in_l in img_npa:
        new_l += list(in_l)
    new_l = sorted(new_l)
    return new_l[int(len(new_l)/qual)]


def check_depth(new_img):
    for val in set(np.array(new_img).ravel()):
        if val%25 != 0:
            return False

    return True


def assert_depth(img):
    img_npa = np.array(img)
    new_npa = np.zeros(img_npa.shape)

    for x in range(img_npa.shape[1]):
        for y in range(img_npa.shape[0]):
            new_val = int(img_npa[y, x] / 25) * 25
            new_npa[y, x] = new_val

    return Image.fromarray(new_npa)


def get_fliped_img(img):
    return Image.fromarray(np.fliplr(np.array(img)))


def reduce_depth(img, given_qual=4):
    img_npa = np.array(img)
    new_npa = np.zeros(img_npa.shape)

    cal_thre = cal_thred(img_npa, qual=given_qual)

    for x in range(img_npa.shape[1]):
        for y in range(img_npa.shape[0]):
            new_val = int(img_npa[y, x] / 25) * 25
            if new_val > cal_thre:
                new_npa[y, x] = 250
            else:
                new_npa[y, x] = new_val

    return Image.fromarray(new_npa)


def sobel(img):
    '''
    Detects edges using sobel kernel
    '''
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)  # detects horizontal edges
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)  # detects vertical edges
    # combine both edges
    return cv2.bitwise_or(opImgx, opImgy)  # does a bitwise OR of pixel values at each pixel


def sketch(frame):
    # Blur it to remove noise
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    # Detect edges from the input image and its negative
    edgImg0 = sobel(frame)
    edgImg1 = sobel(255 - frame)

    edgImg = cv2.addWeighted(edgImg0, 1, edgImg1, 1, 0)  # different weights can be tried too

    # Invert the image back
    opImg = 255 - edgImg
    return opImg


def resize_animism_sketch(img, tar_shape_l=[660, 270]):
    resized_img = resizeimage.resize_contain(img, tar_shape_l, bg_color=(255, 255, 255)).convert('RGB')
    new_img = assert_depth(resized_img.convert('L')).convert('RGB')
    return new_img

def resize_animism_real(img, tar_shape_l=[660, 270]):
    resized_img = resizeimage.resize_contain(img, tar_shape_l, bg_color=(255, 255, 255)).convert('RGB')
    return resized_img


def get_sizes_in_dir(in_dir):
    wth_l = []
    width_l = []
    height_l = []
    for img_na in os.listdir(in_dir):
        img_pa = os.path.join(in_dir, img_na)
        height, width= np.array(Image.open(img_pa)).shape[:2]

        width_l.append(width)
        height_l.append(height)
        wth_l.append(width / height)

    mean_wth = np.mean(wth_l)
    tar_width = sorted(width_l)[-int(len(width_l) / 2)]
    tar_shape = [tar_width, int(tar_width/mean_wth)]
    return mean_wth, tar_width, tar_shape

def resize_tar_dir_pics(in_dir, out_dir):
    mean_wth, tar_width, tar_shape = get_sizes_in_dir(in_dir)
    print('suggest width to height ratio:', mean_wth)
    for img_na in os.listdir(in_dir):
        img_pa = os.path.join(in_dir, img_na)
        img = Image.open(img_pa)
        ori_h, ori_w = np.array(img).shape[:2]
        t_wth = ori_w/ori_h
        tar_shape = [int(ori_h*mean_wth),ori_h] if t_wth < mean_wth else [ori_w, int(ori_w/mean_wth)]

        resized_img = resizeimage.resize_contain(img, tar_shape, bg_color=(255, 255, 255)).convert('RGB')
        resized_img.save(os.path.join(out_dir, img_na))
