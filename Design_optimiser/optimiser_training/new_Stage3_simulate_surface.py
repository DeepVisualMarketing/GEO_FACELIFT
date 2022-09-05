import torch
from Design_optimiser.my_utils import init_trained_rnn, prepare_input_matx


def simulate(rnn, seq_len=10, aes_rise=0.5):
    hidden_status = torch.zeros(1, 1, 8)
    seq_rec_l = []
    input_matx = prepare_input_matx(seq_len=seq_len, init_ms_ratio=0.8, avg_aes=0.8, rise_year=rise_year, aes_rise=aes_rise)
    seq_rec_l.append(float(input_matx[0, 0, 0].data))

    for t_year in range(seq_len-1):

        rst, hidden_status = rnn(input_matx[t_year:t_year+1], h0=hidden_status)
        seq_rec_l.append(float(rst[0, 0, 0].data))
        input_matx[t_year+1, 0, 0] = rst[0, 0, 0] # update the input log-return

    return seq_rec_l

saved_fpa = r'/home/tech/Workspace/Projects/Facelift/optimiser/optimiser/ws_dstore/rnn_model_epoch170_with8hn'
t_rnn = init_trained_rnn(saved_fpa)

simulate(t_rnn)
