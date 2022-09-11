import os
import sys
import numpy as np
from Design_optimiser.DVMCombiner import DVMCombiner


def read_img_rating_rst(saved_rst_fpa):
    rst_d = {}
    cont_d = np.load(saved_rst_fpa, allow_pickle=True).item()
    for idx in cont_d:
        for fna, rating in cont_d[idx]:
            if fna.split('.')[0] not in rst_d:
                rst_d[fna.split('.')[0]] = []
            rst_d[fna.split('.')[0]].append(rating)

    for key in rst_d:
        rst_d[key] = np.mean(rst_d[key])

    return rst_d


def avg_design_rating(img_rating_d, img_d):
    rst_d = {}
    for img_fna in img_rating_d:
        if img_fna not in img_d: continue
        genid = img_d[img_fna]
        year = int(img_fna.split('$$')[2])
        if genid not in rst_d:
            rst_d[genid] = {}
        if year not in rst_d[genid]:
            rst_d[genid][year] = []

        rst_d[genid][year].append(img_rating_d[img_fna])

    new_rst_d = {}
    for genid in rst_d:
        new_rst_d[genid] = []
        for year in range(2008, 2019):
            if year in rst_d[genid]:
                new_rst_d[genid].append((np.mean(rst_d[genid][year]), len(rst_d[genid][year])))
            else:
                new_rst_d[genid].append(None)
    return new_rst_d


def detect_repump(t_l):
    decrease = False
    detect_points_l = []
    for idx, ele in enumerate(t_l):
        if ele is not None and ele < 1: decrease = True
        if decrease and ele is not None and ele > 1.3:
            detect_points_l.append(idx)
            decrease = False
    return detect_points_l


def detect_sales_reboom_points(dvm_combiner, sales_thre = 5000):
    for genid in dvm_combiner.rst_d:
        if 'market_share_changes' in dvm_combiner.rst_d[genid] and np.sum(
                dvm_combiner.rst_d[genid]['sales']) > sales_thre:
            dvm_combiner.rst_d[genid]['reboom'] = detect_repump(dvm_combiner.rst_d[genid]['market_share_changes'])


def cal_aes_changes(dvm_combiner, genid, repump_idx):
    year_id = repump_idx -1 # Note: the market share idx is less 1 than the year idx
    pre_idx_adjust = -1
    post_adjust = 2
    pic_num_thre = 5

    pre_l = [ele for ele in
             dvm_combiner.rst_d[genid]['aes'][max(repump_idx - 3, 0):repump_idx + pre_idx_adjust] if
             ele is not None]
    post_l = [ele for ele in dvm_combiner.rst_d[genid]['aes'][repump_idx:repump_idx + post_adjust] if
              ele is not None]
    if len(pre_l) == 0 or post_l == 0: return None

    pre_pic_sum = np.sum([ele[1] for ele in pre_l if ele is not None])
    post_pic_sum = np.sum([ele[1] for ele in post_l if ele is not None])

    avg_pre_rating = np.sum([ele[0] * ele[1] for ele in pre_l]) / pre_pic_sum

    avg_post_rating = np.sum([ele[0] * ele[1] for ele in post_l]) / post_pic_sum

    if pre_pic_sum < pic_num_thre or post_pic_sum < pic_num_thre or \
            np.isnan(avg_post_rating) or np.isnan(avg_pre_rating):
        return None
    return avg_post_rating - avg_pre_rating


def prepare_sales_data(dvm_combiner):
    for genid in dvm_combiner.rst_d:
        if 'reboom' in dvm_combiner.rst_d[genid] and len(dvm_combiner.rst_d[genid]['reboom']) > 0:
            new_reboom_l = []
            for repump_idx in dvm_combiner.rst_d[genid]['reboom']:
                if 'aes' not in dvm_combiner.rst_d[genid]:
                    aes_change = None
                else:
                    aes_change = cal_aes_changes(dvm_combiner, genid, repump_idx)

                new_reboom_l.append((repump_idx, aes_change))
            dvm_combiner.rst_d[genid]['reboom'] = new_reboom_l


def get_model_avg_design(aes_l):
    if len([tup[1] for tup in aes_l if tup is not None]) < 5 : return None
    sum_aes = np.sum([tup[0] * tup[1] for tup in aes_l if tup is not None])
    sum_imgs = np.sum([tup[1] for tup in aes_l if tup is not None])
    return sum_aes/sum_imgs


def add_zeros_rows(cont_d):
    for model in cont_d:
        if cont_d[model][-1,0] == 0:
            new_row = np.zeros((1, cont_d[model].shape[1]))
            new_row[0, 1] = cont_d[model][-1, 1]
            cont_d[model] = np.concatenate((cont_d[model], new_row,new_row,new_row))

    return cont_d


def convert_to_log_returns(cont_d):
    for model in cont_d:
        for i in range(cont_d[model].shape[0]):
            cont_d[model][i,0] = np.log(max(0.1, cont_d[model][i,0]))

    return cont_d


def merge_as_a_matx(tar_variables_l, ms_change_l):
    # -- Output explain: year1_ms/year2_ms, detected_change_of_aes, undetected_change_of_aes

    t_data_matrx = np.stack([np.array(vari) for vari in tar_variables_l], axis=1)
    slice_idx_l = [idx for idx, val in enumerate(ms_change_l) if val is not None]
    if list(range(slice_idx_l[0], slice_idx_l[-1] + 1)) != slice_idx_l: return None

    t_data_matrx = t_data_matrx[slice_idx_l[0]:slice_idx_l[-1] + 1, :]

    # add 0 market shares to withdraw models
    if t_data_matrx[-1, 0] == 0:
        new_matx = np.zeros((5, t_data_matrx.shape[1]))
        # for i in range(new_matx.shape[0]): new_matx[i, 0] = 1
        t_data_matrx = np.concatenate((t_data_matrx, new_matx))

    return t_data_matrx


basic_tab_fpa = '/home/tech/Workspace/Projects/Facelift/DVM_data_tables/shared_basic_table.csv'
sales_table_pa = '/home/tech/Workspace/Projects/Facelift/DVM_data_tables/shared_sales.csv'
price_tab_fpa = '/home/tech/Workspace/Projects/Facelift/DVM_data_tables/shared_price_table.csv'
ad_tab_fpa = '/home/tech/Workspace/Projects/Facelift/DVM_data_tables/shared_adv_table.csv'
image_tab_fpa = '/home/tech/Workspace/Projects/Facelift/DVM_data_tables/Image_table.csv'


dvm_combiner = DVMCombiner(basic_tab_fpa)
dvm_combiner.add_sale_table(sales_table_pa)
dvm_combiner.add_price_tab(price_tab_fpa)
dvm_combiner.add_bodytype(ad_tab_fpa)
dvm_combiner.cal_sales_changes()
dvm_combiner.cal_market_shares()
dvm_combiner.cal_market_share_changes()
dvm_combiner.detect_sales_reboom_points()

img_d = dvm_combiner.read_img_tab(image_tab_fpa)
img_pred_d = read_img_rating_rst(r'/home/tech/Workspace/Projects/Facelift/optimiser/optimiser/ws_dstore/eval_on_filter_80_via_n_folder.npy')
dvm_combiner.assign_aesthetic_ratings(avg_design_rating(img_pred_d, img_d))
prepare_sales_data(dvm_combiner)
sales_thre = 5000

rst_d = {}

for genid in dvm_combiner.rst_d:

    if 'reboom' in dvm_combiner.rst_d[genid] and len(dvm_combiner.rst_d[genid]['reboom'])>0 and \
        np.sum(dvm_combiner.rst_d[genid]['sales']) > sales_thre and 'market_share_changes' in dvm_combiner.rst_d[genid]:

        aes_val_l, aes_change_known, aes_change_unknown  = np.zeros(10), np.zeros(10), np.zeros(10)

        # Get the avg aes of model over years
        if 'aes' in dvm_combiner.rst_d[genid]:
            avg_des_over_years = get_model_avg_design(dvm_combiner.rst_d[genid]['aes'])
            if avg_des_over_years is not None:
                aes_val_l = np.ones(10)*avg_des_over_years

        for idx, t_tup in enumerate(dvm_combiner.rst_d[genid]['reboom']): # Create the aes change array
            year_idx, val = t_tup
            if val is not None:
                aes_change_known[year_idx-1] = val
            else:
                aes_change_unknown[year_idx-1] = 1

        ms_change_l = list(dvm_combiner.rst_d[genid]['market_share_changes'])

        tar_variables_l = [ms_change_l, aes_change_known, aes_change_unknown]  # aes_val_l,
        t_data_matrx = merge_as_a_matx(tar_variables_l, dvm_combiner.rst_d[genid]['market_share_changes'])
        if t_data_matrx is None: continue

        rst_d[dvm_combiner.rst_d[genid]['Automaker']+'_'+dvm_combiner.rst_d[genid]['Genmodel']] = t_data_matrx

# np.save('ws_dstore/data_for_rnn_training_v3',rst_d)
