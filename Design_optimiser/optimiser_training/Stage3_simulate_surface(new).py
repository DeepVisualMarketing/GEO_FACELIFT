import os
import torch
from Design_optimiser.my_utils import MYRNN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from argparse import ArgumentParser
from Design_optimiser.my_utils import convert_ratio_to_ms


def simulate_diff_initial_ms_surface():
    seq_len = 10
    x_l, y_l, z_l = [],[],[]
    init_ms_ratio = 0.15
    avg_aes = 1.5
    t_range = [val/10 for val in range(-5, 10)]
    rise_year = 3
    aes_rise = 0
    for init_ms_ratio in t_range:
        hidden_status = torch.zeros(1, 1, 8)
        seq_rec_l = []
        t_input = np.array([init_ms_ratio, avg_aes, 0, 0]).reshape((1, 1, 4))
        t_input = torch.from_numpy(t_input).float()

        for t_year in range(seq_len):
            if t_year == rise_year:
                t_input[0, 0, 2] = aes_rise
            else:
                t_input[0, 0, 2] = 0

            seq_rec_l.append(float(t_input[0, 0, 0].detach().cpu().numpy()))
            rst, hidden_status = rnn(t_input, h0=hidden_status)
            t_input[0, 0, 0] = rst[0, 0, 0] # update the input log-return
        x_l.append(list(range(len(seq_rec_l))))
        z_l.append([np.power(np.e, val) for val in seq_rec_l])
        print(seq_rec_l)

    y_l = [list(t_range) for i in range(len(seq_rec_l))]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.array(x_l)
    Y = np.transpose(np.array(y_l))
    Z = np.array(z_l)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('Years')
    ax.set_ylabel('Aes Rise')
    ax.set_zlabel('Ratio')

    plt.show()


def simulate_diff_aes_rise_ms_surface(args):
    seq_len = 30
    x_l, y_l, z_l,h_l = [],[],[],[]
    init_ms_ratio = 0.8 #0.15
    t_range = [val/100 for val in range(-20, 20)]
    rise_year = 10
    for aes_rise in t_range:
        hidden_status = torch.zeros(1, 1, args.hidden_dim)
        seq_rec_l = []
        t_input = np.array([init_ms_ratio, 0, 0]).reshape((1, 1, 3))
        t_input = torch.from_numpy(t_input).float()

        for t_year in range(seq_len):
            if t_year == rise_year:
                t_input[0, 0, 1] = aes_rise
            else:
                t_input[0, 0, 1] = 0
            seq_rec_l.append(float(t_input[0, 0, 0].detach().cpu().numpy()))
            rst, hidden_status = rnn(t_input, h0=hidden_status)
            t_input[0, 0, 0] = rst[0, 0, 0]
        h_l.append(convert_ratio_to_ms(seq_rec_l))


    x_l = [list(range(len(seq_rec_l))) for i in t_range]
    y_l = [list(t_range) for i in seq_rec_l]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.array(x_l)
    Y = np.transpose(np.array(y_l))
    H = np.array(h_l)
    ax.plot_surface(X, Y, H, cmap=cm.coolwarm)

    ax.set_xlabel('Years')
    ax.set_ylabel('Aesthetic change')
    ax.set_zlabel('Market share')

    plt.show()


def simulate_diff_rise_year_ms_surface(args):
    seq_len = 10
    x_l, y_l, z_l,h_l = [],[],[],[]
    init_ms_ratio = 0.8
    t_range = [val for val in range(1, 9)]
    aes_rise = 0.1
    for rise_year in t_range:
        hidden_status = torch.zeros(1, 1, args.hidden_dim)
        seq_rec_l = []
        t_input = np.array([init_ms_ratio, 0, 0]).reshape((1, 1, 3))
        t_input = torch.from_numpy(t_input).float()

        for t_year in range(seq_len):
            if t_year == rise_year:
                t_input[0, 0, 1] = aes_rise
            else:
                t_input[0, 0, 1] = 0
            seq_rec_l.append(float(t_input[0, 0, 0].detach().cpu().numpy()))
            rst, hidden_status = rnn(t_input, h0=hidden_status)
            t_input[0, 0, 0] = rst[0, 0, 0] # update the input log-return
        x_l.append(list(range(len(seq_rec_l))))
        _lg_return_l = [val for val in seq_rec_l]
        z_l.append(_lg_return_l)
        h_l.append(convert_ratio_to_ms(_lg_return_l))
        print(seq_rec_l)

    y_l = [list(t_range) for i in range(len(seq_rec_l))]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.array(x_l)
    Y = np.transpose(np.array(y_l))
    Z = np.array(z_l)
    H = np.array(h_l)
    ax.plot_surface(X, Y, H, cmap=cm.coolwarm)

    ax.set_xlabel('Years')
    ax.set_ylabel('Facelift year')
    ax.set_zlabel('Market share')

    plt.show()

args = ArgumentParser().parse_args()
args.start_dim, args.end_dim = 0, 3
args.hidden_dim = 8   # r1:8 / r2:6
INPUT_SIZE = args.end_dim - args.start_dim
rnn = MYRNN(INPUT_SIZE, output_size=1, hidden_dim=args.hidden_dim, n_layers=1)

saved_fpa = r'/home/tech/Workspace/Projects/Facelift/optimiser/optimiser/ws_dstore/rnn_model_epoch350'
rnn.load_state_dict(torch.load(saved_fpa))

simulate_diff_aes_rise_ms_surface(args)
# simulate_diff_rise_year_ms_surface(args)
