import torch, sys
import numpy as np
from torch import nn
from random import shuffle
import torch.nn.init as init
from Design_optimiser.my_utils import MYRNN
np.random.seed(17)
torch.manual_seed(17)


def train(rnn, train_l, test_l, stat_d, out_dir, start_dim=2, end_dim=4):
    criterion = nn.MSELoss()
    test_sum_loss = 0
    test_loss_rec_l = []
    for epoch in range(2000):
        shuffle(train_l)
        sum_loss = 0
        for iter_idx, t_batch in enumerate(train_l):
            x = torch.from_numpy(t_batch[:, :-1, start_dim:end_dim].astype('float')).float()  #:-interval_len
            y = torch.from_numpy(t_batch[:, 1:, 0:1].astype('float')).float() #interval_len:

            output,hidden_out = rnn(x)
            loss = criterion(output, y)
            sum_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 and epoch > 0:
            for t_batch in test_l:
                x = torch.from_numpy(t_batch[:, :-1, start_dim:end_dim].astype('float')).float()
                y = torch.from_numpy(t_batch[:, 1:, 0:1].astype('float')).float()

                output, hidden_out = rnn(x)
                loss = criterion(output,y)
                test_sum_loss += loss

            if epoch % 5 == 0:
                print(f'Epoch {epoch} Loss: {np.round(float(sum_loss.data),2)} + Test Loss: {np.round(float(test_sum_loss.data), 2)}')
                test_loss_rec_l.append(np.round(test_sum_loss.detach().numpy(), 2))
                test_sum_loss = 0

            torch.save(rnn.state_dict(), f'{out_dir}/rnn_model_epoch{epoch}')
    return test_loss_rec_l


def assign_to_batchs(cont_d, batch_size=4):
    data_d = {}
    for key in cont_d:
        t_len = cont_d[key].shape[0]
        if t_len not in data_d:
            data_d[t_len] = []
        data_d[t_len].append(cont_d[key])

    train_l, test_l = [], []
    for t_len in data_d:
        for start_idx in range(0, len(data_d[t_len]), batch_size):
            t_batch = np.array(data_d[t_len][start_idx:start_idx+batch_size])
            if np.random.rand() < 1.0/4:
                test_l.append(t_batch)
            else:
                train_l.append(t_batch)
    return train_l, test_l

def orthogonal_init_model(t_model):
    for param in t_model.parameters():
        if len(param.shape) >= 2:
            init.orthogonal_(param.data)
        else:
            init.normal_(param.data)
    return t_model

if __name__ == '__main__':
    INPUT_SIZE=None
    start_dim, end_dim = 0, 3
    INPUT_SIZE = end_dim - start_dim
    cont_d = np.load(r'/home/tech/Workspace/Projects/Facelift/optimiser/optimiser/ws_dstore/data_for_rnn_training_v3.npy',
                     allow_pickle=True).item()
    stat_d = {}
    out_dir = '/home/tech/Workspace/Projects/Facelift/optimiser/optimiser/ws_dstore/training_r3'

    for batch_size in [4]:
        cont_l = list(cont_d.values())
        train_l, test_l = assign_to_batchs(cont_d, batch_size=batch_size)
        print('length of test', len(test_l))
        print('train_size', len(train_l), 'test size', len(test_l))
        for wr in [0.0000001]:
            for lr in [0.005]:
                for hidden_neuro_num in [8]:

                    setting_key = f'batch-{batch_size}lr-{lr}hn{hidden_neuro_num}wr-{wr}'
                    rnn = MYRNN(INPUT_SIZE, output_size=1, hidden_dim=hidden_neuro_num, n_layers=1)
                    rnn = orthogonal_init_model(rnn)
                    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=0.0000001)
                    stat_d[setting_key] = train(rnn, train_l, test_l,stat_d, out_dir, start_dim, end_dim)

