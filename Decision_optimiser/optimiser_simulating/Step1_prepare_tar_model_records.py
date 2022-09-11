import sys
import numpy as np
from Design_optimiser.DVMCombiner import DVMCombiner


def avg_design_rating(img_rating_d, match_d):
    rst_d = {}
    for img_fna in img_rating_d:
        maker, model = img_fna.split('$$')[:2]
        if (model, maker) not in match_d: continue

        genid = match_d[(model, maker)]
        year = int(img_fna.split('$$')[2])
        if genid not in rst_d:
            rst_d[genid] = {}
        if year not in rst_d[genid]:
            rst_d[genid][year] = []

        rst_d[genid][year].append(img_rating_d[img_fna])

    new_rst_d = {}
    for genid in rst_d:
        new_rst_d[genid] = []
        for year in range(2001, 2019):
            if year in rst_d[genid]:
                new_rst_d[genid].append((np.mean(rst_d[genid][year]), len(rst_d[genid][year])))
            else:
                new_rst_d[genid].append(None)
    return new_rst_d


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


basic_tab_fpa = '/home/tech/Workspace/Projects/DVM/New_DVM/Phase3___Prepare_shared_dataset/DVM-Car_V2.0/Basic_table (reduced_genmodel_table).csv'
sales_table_pa = '/home/tech/Workspace/Projects/DVM/New_DVM/Phase3___Prepare_shared_dataset/DVM-Car_V2.0/Sales_table.csv'
price_tab_fpa = '/home/tech/Workspace/Projects/DVM/New_DVM/Phase3___Prepare_shared_dataset/DVM-Car_V2.0/Price_table.csv'
ad_tab_fpa = '/home/tech/Workspace/Projects/DVM/New_DVM/Phase3___Prepare_shared_dataset/DVM-Car_V2.0/Ad_table.csv'
image_tab_fpa = '/home/tech/Workspace/Projects/DVM/New_DVM/Phase3___Prepare_shared_dataset/DVM-Car_V2.0/Image_table.csv'
tar_l = ['7_5', '95_18', '69_26', '95_21', '69_5', '80_10', '92_34', '29_4', '94_5', '8_1', '34_11', '36_8']
pred_rst_fpa = '/home/tech/Workspace/Projects/Facelift/optimiser/optimiser/ws_dstore/eval_on_filter_80_via_n_folder.npy'

dvm_combiner = DVMCombiner(basic_tab_fpa,start_year=2001, end_year=2018)
dvm_combiner.add_sale_table(sales_table_pa)
dvm_combiner.add_price_tab(price_tab_fpa)
dvm_combiner.add_bodytype(ad_tab_fpa)
dvm_combiner.cal_market_shares()
dvm_combiner.cal_sales_changes()
dvm_combiner.cal_market_share_changes()
img_pred_d = read_img_rating_rst(pred_rst_fpa)
dvm_combiner.assign_aesthetic_ratings(avg_design_rating(img_pred_d, dvm_combiner.match_d))

rst_d = {}
for gid in tar_l:
    model_key = (dvm_combiner.rst_d[gid]['Automaker'], dvm_combiner.rst_d[gid]['Genmodel'])
    rst_d[model_key] = {'sales': dvm_combiner.rst_d[gid]['sales'],
                           'market_shares': dvm_combiner.rst_d[gid]['market_shares'],
                            'aes': dvm_combiner.rst_d[gid]['aes'],
                           'market_share_changes': dvm_combiner.rst_d[gid]['market_share_changes'],
                           'market_share_changes_log': dvm_combiner.rst_d[gid]['market_share_changes_log'],}




