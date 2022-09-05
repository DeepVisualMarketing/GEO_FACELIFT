# @Author : JM_Huang
# @Time   : 02/11/19
import numpy as np
import torch
from torch import nn


def get_sales_changes(later_y, this_y):
    this_y = float(this_y)
    later_y = float(later_y)

    if this_y < 100:
        return -1
    else:
        # change = (later_y - this_y)/float(this_y)
        change = (later_y) / float(this_y)

        return min(change, 5.0)


def convert_ratio_to_ms(ratio_l, true_ms=1):
    new_l = []
    for val in ratio_l:
        true_ms = true_ms * (float(val.data) if isinstance(val, torch.Tensor) else val)
        new_l.append(true_ms)
    return new_l


def read_sale_table(f_pa):
    sales_d = {}
    sales_change_d = {}
    gen_id_d = {}

    with open(f_pa) as f_in:
        for line in f_in.readlines()[1:]:
            pieces = line.strip().split(',')
            model_na = pieces[1].lower()
            ID = pieces[2]
            sales_d[ID] = [int(float(val)) for val in pieces[13:2:-1]] # 2008:2018 sales
            gen_id_d[ID] = model_na
            # if np.sum(sales_d[ID])<5000 or np.max(sales_d[ID])<5000: continue

            sales_change_d[ID] = [get_sales_changes(sales_d[ID][idx + 1], sales_d[ID][idx]) for idx, val in
             enumerate(sales_d[ID]) if idx < len(sales_d[ID]) - 1]

    new_change_d = {}
    for key in sales_change_d:
        if len([ele for ele in sales_change_d[key] if ele is not None]):
            new_change_d[key] = sales_change_d[key]
    return sales_d, new_change_d, gen_id_d


def get_model_avg_ratings(t_npa, img_d, tar_peri=(2013,2017)):
    cont_d = {}
    for row in t_npa:
        year=int(row[0].split('$$')[2])
        if year > tar_peri[1] or year < tar_peri[0] or row[0] not in img_d: continue
        t_key = img_d[row[0]]
        if t_key not in cont_d:
            cont_d[t_key] = []
        cont_d[t_key].append(float(row[1]))

    for key in cont_d:
        cont_d[key] = np.mean(cont_d[key])
    return cont_d


def avg_ratings(t_npa, by_year=False):
    cont_d = {}
    for row in t_npa:
        pieces = row[0].lower().split('$$')
        maker, model, year = pieces[:3]
        key = (maker, model, year) if by_year else (maker, model)
        if key not in cont_d:
            cont_d[key] = []
        cont_d[key].append(float(row[1]))

    for key in cont_d:
        cont_d[key] = np.mean(cont_d[key])
    return cont_d


def combine_aes_rating_with_sales(gen_id_d, aes_d, sales_d, price_d, bodytype_d):
    cont_d = {}
    for t_key in sales_d:
        complete = True
        for idx, tar_d in enumerate([gen_id_d, aes_d, price_d, bodytype_d]):
            if t_key not in tar_d:
                # print(f'{t_key} not in {idx} dict')
                complete = False
        if not complete: continue
        cont_d[t_key] = {'name':gen_id_d[t_key], 'aes':aes_d[t_key], 'sales':sales_d[t_key], 'price': price_d[t_key],
                         'bodytype': bodytype_d[t_key]}
    return cont_d


def add_aes_percent(avg_pred_d):
    sorted_l = sorted(list(avg_pred_d.values()))
    for key in avg_pred_d:
        avg_pred_d[key] = (avg_pred_d[key], sorted_l.index(avg_pred_d[key])/len(sorted_l))
    return avg_pred_d


def avg_tar_year_ratings(avg_pred_d, tar_peri=(2013, 2018)):
    cont_d = {}
    for key in avg_pred_d:
        maker, model, year = key
        if int(year) < tar_peri[0] or int(year) > tar_peri[1]: continue

        if (maker, model) not in cont_d:
            cont_d[(maker, model)] = []
        cont_d[(maker, model)].append(avg_pred_d[key])

    new_cont_d = dict([(item[0], np.mean(item[1])) for item in cont_d.items()])
    return new_cont_d


def get_body_type_d(t_fpa):
    cont_d = {}
    with open(t_fpa) as f_in:
        for line in f_in.readlines()[1:]:
            pieces = line.strip().split(',')
            if pieces[7] == 'Unko': continue
            gen_id, year, bodytype = pieces[2], int(pieces[7]), pieces[8]
            if year < 2013: continue
            if gen_id not in cont_d:
                cont_d[gen_id] = {}
            if bodytype not in cont_d[gen_id]:
                cont_d[gen_id][bodytype] = 0

            cont_d[gen_id][bodytype] += 1

    for gen_id in cont_d:
        cont_d[gen_id] = sorted(list(cont_d[gen_id].items()), key=lambda x: x[1])[-1]

    return cont_d


def read_img_tab(image_tab_fpa):
    cont_d = {}
    with open(image_tab_fpa) as f_in:
        for line in f_in.readlines()[1:]:
            pieces = line.strip().split(',')
            genid, pic_na = pieces[:2]
            cont_d['$$'.join(pic_na.split('$$')[:4] + pic_na.split('$$')[5:])] = genid
    return cont_d


def init_trained_rnn(saved_fpa, start_dim=0, end_dim=4, hidden_dim=8, n_layers=1, output_size=1):
    INPUT_SIZE = end_dim - start_dim
    rnn = MYRNN(INPUT_SIZE, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)
    rnn.load_state_dict(torch.load(saved_fpa))

    return rnn


def prepare_input_matx(pre_ms_log_l=[0.8], known_aes_rise_l=None,unknown_aes_rise_l=None, start_year=None):
    attr_num = 4

    t_input = np.zeros((len(pre_ms_log_l),1, attr_num))
    t_input[0:len(pre_ms_log_l),0, 0] = pre_ms_log_l
    # t_input[:, 0, 1] = avg_aes

    if start_year is not None:
        t_input[:, 0, -1] = list(range(start_year, len(pre_ms_log_l)+start_year))

    if known_aes_rise_l is not None:
        for idx, val in known_aes_rise_l:
            t_input[idx, 0, 1] = val
    if unknown_aes_rise_l is not None:
        for idx in unknown_aes_rise_l:
            t_input[idx, 0, 2] = 1

    t_input = torch.from_numpy(t_input).float()

    return t_input


class MYRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(MYRNN, self).__init__()
        self.layer_n = n_layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, h0=None, print_hidden=False):
        if h0 is None:
            h0 = torch.zeros(self.layer_n, x.size(0), self.hidden_dim)
        # c0 = torch.zeros(self.layer_n, x.size(0), self.hidden_dim)
        all_h, _ = self.rnn(x, h0)

        delay_num = 0
        r_out = all_h[:, delay_num:, :]  # Here the delay num determine how long the interval between the input and output
        if print_hidden: print(r_out)
        output = self.fc(r_out)

        return output, all_h
