import os
import sys
import torch
import numpy as np
from Design_optimiser.optimser_simulating.my_utils import init_trained_rnn, prepare_input_matx, convert_ratio_to_ms
from argparse import ArgumentParser
from Design_optimiser.optimser_simulating.Step2_1_simulation_paras import *


def simulate_rnn(rnn, input_matx, matx_to_fill):
    hidden_status = torch.zeros(1, 1, 8)
    rst = None
    for t_idx in range(input_matx.shape[0]):
        rst, hidden_status = rnn(input_matx[t_idx:t_idx + 1], h0=hidden_status)
    matx_to_fill[0, 0, 0] = float(rst.data)
    for t_idx in range(0, matx_to_fill.shape[0]-1):
        rst, hidden_status = rnn(matx_to_fill[t_idx:t_idx + 1], h0=hidden_status)
        matx_to_fill[t_idx+1, 0, 0] = float(rst.data)  # update the input

    return matx_to_fill


def obtain_aes_scores(audi_cont_d):
    aes_raw_l = [(idx + 2001, val) for idx, val in enumerate(audi_cont_d['aes'])]
    avg_aes = np.sum([val[0]*val[1] for val in aes_raw_l]) / np.sum([val[1] for val in aes_raw_l])


def extract_tar_piece(t_tup_l, min_year, max_year):
    return [val for year, val in t_tup_l if year>=min_year and year<=max_year]


def simulate(t_rnn, ms_change_for_input, start_year, face_lift_year, end_year, rise_lvl = 0.09, print_matx=False):
    whole_len = end_year - start_year + 1
    facelift_pos = face_lift_year - start_year
    aes_rise_l1, aes_rise_l2 = [], []
    if facelift_pos < len(ms_change_for_input):
        aes_rise_l1.append((facelift_pos, rise_lvl))
    else:
        aes_rise_l2.append((facelift_pos-len(ms_change_for_input), rise_lvl))
    input_matx = prepare_input_matx(pre_ms_log_l=ms_change_for_input,  known_aes_rise_l=aes_rise_l1, start_year=start_year)
    matx_to_fill = prepare_input_matx(pre_ms_log_l=[0]*int(whole_len- len(ms_change_for_input)), known_aes_rise_l=aes_rise_l2, start_year=start_year+len(ms_change_for_input))

    rst_matx = simulate_rnn(t_rnn, input_matx[:,:,:3], matx_to_fill[:,:,:3])
    if print_matx:
        print('input matx', input_matx)
        print('rst matx', matx_to_fill)
    return matx_to_fill


def display_ms_changes(in_ms_l):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 2))
    ax1 = fig.add_subplot()
    ax1.step([year for year, val in in_ms_l], [val for year, val in in_ms_l])
    plt.show()


def simulate_original_plan(rst_d, t_rnn, args, ms_change_for_input):

    matx_to_fill = simulate(t_rnn, ms_change_for_input, args.simulation_start_year-args.input_pre_extend, args.ori_facelift_year,
                           args.simulation_end_year, rise_lvl = args.ori_facelift_aes-args.base_aes, print_matx=False)
    t_ms_l = convert_ratio_to_ms(matx_to_fill[:, 0, 0], true_ms=args.start_ms)
    rst_d[args.genmodel][('Original facelift', args.ori_facelift_aes, args.ori_facelift_year)] = [np.sum(t_ms_l), t_ms_l]

    return rst_d


def simulate_given_facelifts(rst_d, given_aes_rises_l, t_rnn, args, ms_change_for_input):

    for idx, aes_rise in enumerate(given_aes_rises_l):
        max_one_l = [0, None, None, None]
        for face_lift_year in range(args.simulation_start_year, args.simulation_end_year):
            matx_to_fill = simulate(t_rnn, ms_change_for_input, args.simulation_start_year-args.input_pre_extend, face_lift_year,
                                    args.simulation_end_year, rise_lvl = aes_rise, print_matx=False)
            t_ms_l = convert_ratio_to_ms(matx_to_fill[:, 0, 0], true_ms=args.start_ms)
            if np.sum(t_ms_l) > max_one_l[0]:
                max_one_l = [np.sum(t_ms_l), aes_rise, face_lift_year, t_ms_l]

        # print(
        #     f'Predicted:: New Aes {max_one_l[1]} Year{max_one_l[2]} Sum {round(max_one_l[0], 7)} Details{max_one_l[3]}')

        rst_d[args.genmodel][(f'New design{idx+1}', aes_rise, max_one_l[2])] = [np.sum(max_one_l[3]), max_one_l[3]]
    return rst_d


def simulate_given_model(args):
    cont_d = np.load(args.saved_ms_fpa, allow_pickle=True).item()[(args.maker, args.genmodel)]
    t_rnn = init_trained_rnn(args.saved_rnn_fpa, end_dim=3)
    whole_ms_l = [(args.base_year+idx, val*100) for idx, val in enumerate(cont_d['market_shares'])]
    whole_ms_change_l = [(args.base_year+idx+1, val) for idx, val in enumerate(cont_d['market_share_changes'])]

    ms_change_b4_launch = extract_tar_piece(whole_ms_change_l, args.simulation_start_year - args.input_pre_extend,
                                            args.simulation_start_year - 1)
    if len([val for val in ms_change_b4_launch if val is not None])==1:
        ms_change_b4_launch = [1 if val is None else val for val in ms_change_b4_launch]
        ms_change_b4_launch = [val/2 if val > 3 else val for val in ms_change_b4_launch]

    # print('real', extract_tar_piece(whole_ms_change_l, args.simulation_start_year, args.simulation_end_year))
    args.start_ms = extract_tar_piece(whole_ms_l, args.simulation_start_year-1, args.simulation_start_year-1)[0]

    #============================Simulation and save==============================
    rst_d = {args.genmodel:{}}
    rst_d[f'{args.genmodel}_original'] = extract_tar_piece(whole_ms_l, args.simulation_start_year-1, args.simulation_end_year)  # The ground truth
    rst_d[f'{args.genmodel}_args'] = args

    rst_d = simulate_original_plan(rst_d, t_rnn, args, ms_change_b4_launch)
    rst_d = simulate_given_facelifts(rst_d, args.given_aes_rises_l, t_rnn, args, ms_change_b4_launch)
    # np.save(f'ws_dstore/{args.genmodel.lower()}_simulation', rst_d)

    return rst_d

