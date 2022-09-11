import os
import torch
import numpy as np
from numpy import array, linspace
from Design_optimiser.my_utils import MYRNN


def simulate_a_cycle(rnn, init_ms_ratio, avg_aes, rise_year_l, aes_rise):
    seq_rec_l = []
    hidden_status = torch.zeros(1, 1, 8)

    t_input = torch.from_numpy(np.array([init_ms_ratio, avg_aes, 0, 0]).reshape((1, 1, 4))).float()

    for t_year in range(SEQ_LEN):
        t_input[0, 0, 2] = aes_rise if t_year in rise_year_l else 0
        seq_rec_l.append(float(t_input[0, 0, 0].detach().cpu().numpy()))

        rst, hidden_status = rnn(t_input, h0=hidden_status)
        t_input[0, 0, 0] = rst[0, 0, 0]  # update the input log-return

    return seq_rec_l


def convert_to_ms_l(seq_rec_l):
    ratio_l = [np.power(np.e, val) for val in seq_rec_l]
    t_prod = 1
    new_l = []
    for val in ratio_l:
        t_prod = t_prod * val
        new_l.append(t_prod)
    return new_l


def facelift_first_year(rnn, init_ms_log, avg_aes):
    global SEQ_LEN

    rise_year_l = [1]
    aes_rise = 0.008
    fl_seq_rec_l = simulate_a_cycle(rnn, init_ms_log, avg_aes, rise_year_l, aes_rise)
    nonfl_seq_rec_l = simulate_a_cycle(rnn, init_ms_log, avg_aes, [], aes_rise)
    return np.sum(convert_to_ms_l(fl_seq_rec_l))- np.sum(convert_to_ms_l(nonfl_seq_rec_l))


def facelift_every_n_years(rnn, n, init_ms_log, avg_aes):
    global SEQ_LEN

    t_range = [val for val in range(1, 9)]
    aes_rise = 0.5
    rise_year_l = [i for i in range(SEQ_LEN) if i %n ==0 and i>0]
    aes_rise = 0.008*n
    seq_rec_l = simulate_a_cycle(rnn, init_ms_log, avg_aes, rise_year_l, aes_rise)
    # print(np.sum(convert_to_ms_l(seq_rec_l)), convert_to_ms_l(seq_rec_l))

    return np.sum(convert_to_ms_l(seq_rec_l))


# ====================================================================================================

SEQ_LEN = 10
first_ms_change_points = [-1.3847403592531562, -1.1428571428571428, 0, 0.9795918367346939, 2.9972386760482475]
avg_aes_points = [0.6597540428790639, 0.85768774, 1.70928277, 2.01895369, 2.100760795844822]

start_dim, end_dim = 0, 4
INPUT_SIZE = end_dim - start_dim
rnn = MYRNN(INPUT_SIZE, output_size=1, hidden_dim=8, n_layers=1)

saved_fpa = r'D:\OneDrive - University of Glasgow\My_codes\Facelift\optimiser\ws_dstore\rnn_model_epoch170_with8hn'
rnn.load_state_dict(torch.load(saved_fpa))

# ====================================================================================================

start_dim, end_dim = 0, 4
INPUT_SIZE = end_dim - start_dim
rnn = MYRNN(INPUT_SIZE, output_size=1, hidden_dim=8, n_layers=1)

saved_fpa = r'D:\OneDrive - University of Glasgow\My_codes\Facelift\optimiser\ws_dstore\rnn_model_epoch170_with8hn'
rnn.load_state_dict(torch.load(saved_fpa))

with open('out.csv', 'w') as f_out:
    for n in [1,2,3,4]:
        cand_ms_log_l = linspace(first_ms_change_points[1], first_ms_change_points[2], 10)
        avg_aes_points_l = linspace(avg_aes_points[1], avg_aes_points[3], 10)
        sum_val = 0
        for init_ms_log in cand_ms_log_l:
            for avg_aes in avg_aes_points_l:
                sum_val += facelift_every_n_years(rnn, n, init_ms_log, avg_aes=avg_aes)
        mean_val = sum_val/(len(cand_ms_log_l)*len(avg_aes_points_l))
        f_out.write('{},{:.4}\n'.format(n, mean_val))

