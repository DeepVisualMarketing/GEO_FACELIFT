import numpy as np
from copy import deepcopy
from scipy import stats


class DVMCombiner:
    def __init__(self, basic_table_fpa, start_year=2008, end_year=2018):
        self.rst_d = {}
        self.match_d = {}
        self.init_basic_tab(basic_table_fpa)
        self.version=1.0
        self.start_year = start_year
        self.end_year = end_year
        self.seg_market_aes_d = None

    def init_basic_tab(self, basic_table_fpa):
        tab_cont_l, attr_d = self.read_formed_csv(basic_table_fpa)
        for row in tab_cont_l:
            self.rst_d[row[attr_d['Genmodel_ID']]] = {'Genmodel':row[attr_d['Genmodel']], 'Automaker':row[attr_d['Automaker']]}
            self.match_d[(row[attr_d['Genmodel']], row[attr_d['Automaker']] )] = row[attr_d['Genmodel_ID']]

    def read_formed_csv(self, fpa):
        with open(fpa) as f_in:
            cont = f_in.readlines()
            attr_d = dict([(word, idx) for idx, word in enumerate(cont[0].strip().replace(' ','').split(','))])

            return [line.strip().split(',') for line in cont[1:]], attr_d

    def get_sales_changes(self, later_y, this_y):
        this_y = float(this_y)
        later_y = float(later_y)

        if this_y < 100 or later_y < 100:
            return None
        else:
            change = (later_y - this_y) / float(this_y)
            return change

    def add_sale_table(self, tab_fpa):

        tab_cont_l, attr_d = self.read_formed_csv(tab_fpa)

        for row in tab_cont_l:
            gen_id = row[attr_d['Genmodel_ID']]
            t_val = np.array([int(float(val)) for val in row[attr_d[str(self.start_year)]:attr_d[str(self.end_year)] - 1:-1]]) # so 11 values from 2008 to 2018

            if 'sales' in self.rst_d[gen_id]:
                self.rst_d[gen_id]['sales'] = self.rst_d[gen_id]['sales'] + t_val
            else:
                self.rst_d[gen_id]['sales'] = t_val

        for gen_id in self.rst_d:
            if 'sales' in self.rst_d[gen_id]:
                t_sals_npa = self.rst_d[gen_id]['sales']
                self.rst_d[gen_id]['sales_change'] = [self.get_sales_changes(t_sals_npa[idx + 1],
                    t_sals_npa[idx]) for idx, val in enumerate(t_sals_npa) if idx < len(t_sals_npa) - 1]

    def add_price_tab(self, tab_fpa):
        tab_cont_l, attr_d = self.read_formed_csv(tab_fpa)
        for row in tab_cont_l:
            gen_id = row[attr_d['Genmodel_ID']]
            if 'price' not in self.rst_d[gen_id]:
                self.rst_d[gen_id]['price'] = [None] * (self.end_year-self.start_year+1)

            year = int(row[attr_d['Year']])
            if not self.end_year>=year >=self.start_year: continue

            self.rst_d[gen_id]['price'][year-self.start_year] = int(row[attr_d['Entry_price']])

    def add_bodytype(self, tab_fpa):
        cont_d = {}
        tab_cont_l, attr_d = self.read_formed_csv(tab_fpa)

        for row in tab_cont_l:
            if row[attr_d['Reg_year']] == 'Unko':
                continue
            gen_id = row[attr_d['Genmodel_ID']]
            bodytype = row[attr_d['Bodytype']]

            if len(row[attr_d['Reg_year']])==0 or int(row[attr_d['Reg_year']]) < 2007: continue
            if gen_id not in cont_d:
                cont_d[gen_id] = {}
            if bodytype not in cont_d[gen_id]:
                cont_d[gen_id][bodytype] = 0

            cont_d[gen_id][bodytype] += 1

        for gen_id in cont_d:
            self.rst_d[gen_id]['bodytype'] = sorted(list(cont_d[gen_id].items()), key=lambda x: x[1])[-1][0]

    def add_aesthetic(self, aes_pred_d):
        for gen_id in aes_pred_d:
            self.rst_d[gen_id]['aes'] = aes_pred_d[gen_id]

    def cal_bodytype_sum_sales(self):
        self.bodytype_sum_sales_d = {}
        for key in self.rst_d:
            if 'sales' in self.rst_d[key] and 'bodytype' in self.rst_d[key]:
                t_bodytype = self.rst_d[key]['bodytype']
                if t_bodytype not in self.bodytype_sum_sales_d:
                    self.bodytype_sum_sales_d[t_bodytype] = deepcopy(self.rst_d[key]['sales'])
                else:
                    self.bodytype_sum_sales_d[t_bodytype] += deepcopy(self.rst_d[key]['sales'])


    def cal_market_shares(self):
        self.cal_bodytype_sum_sales()

        for key in self.rst_d:
            if 'sales' in self.rst_d[key] and 'bodytype' in self.rst_d[key]:
                t_bodytype = self.rst_d[key]['bodytype']
                self.rst_d[key]['market_shares'] = np.nan_to_num(self.rst_d[key]['sales']/self.bodytype_sum_sales_d[t_bodytype])

    def cal_ms_change_ratio(self, t_pre, t_post):
        if t_pre < 0.0001:
            return None
        else:
            if t_post < 0.0001:
                return 0
            else:
                val = t_post / t_pre
                return val

    def cal_sales_change_ratio(self, t_pre, t_post):
        if t_pre < 100:
            return None
        else:
            if t_post < 100:
                return 0
            else:
                val = t_post / t_pre if t_post / t_pre < 50 else None
                return val

    def cal_sales_changes(self, default_log_max=0.1):
        for key in self.rst_d:
            if 'sales' not in self.rst_d[key]: continue

            self.rst_d[key]['sales_changes'], self.rst_d[key]['sales_changes_log'] = [], []

            for idx, val in enumerate(list(self.rst_d[key]['sales'])):
                if idx == len(self.rst_d[key]['sales'])-1: continue

                val = self.cal_sales_change_ratio(self.rst_d[key]['sales'][idx], self.rst_d[key]['sales'][idx+1])
                self.rst_d[key]['sales_changes'].append(val)
                self.rst_d[key]['sales_changes_log'].append(np.log(max(default_log_max, val)) if val is not None else None)


    def cal_market_share_changes(self, default_log_max=0.1):
        for key in self.rst_d:
            if 'market_shares' not in self.rst_d[key]: continue

            self.rst_d[key]['market_share_changes'], self.rst_d[key]['market_share_changes_log'] = [], []
            for idx, val in enumerate(list(self.rst_d[key]['market_shares'])):
                if idx == len(self.rst_d[key]['sales'])-1: continue

                val = self.cal_ms_change_ratio(self.rst_d[key]['market_shares'][idx], self.rst_d[key]['market_shares'][idx+1])
                self.rst_d[key]['market_share_changes'].append(val)
                self.rst_d[key]['market_share_changes_log'].append(np.log(max(default_log_max, val)) if val is not None else None)


    @staticmethod
    def read_img_tab(image_tab_fpa):
        cont_d = {}
        with open(image_tab_fpa) as f_in:
            for line in f_in.readlines()[1:]:
                pieces = line.strip().split(',')
                genid, pic_na = pieces[:2]
                key = '$$'.join(pic_na.split('$$')[:4] + pic_na.split('$$')[5:])
                cont_d[key.split('.')[0]] = genid
        return cont_d

    def assign_aesthetic_ratings(self, ratings_d):
        for genid in ratings_d:
            self.rst_d[genid]['aes'] = ratings_d[genid]

    def detect_repump(self, t_l):
        decrease = False
        detect_points_l = []
        for idx, ele in enumerate(t_l):
            if ele is not None and ele < 1: decrease = True
            if decrease and ele is not None and ele > 1.1:
                detect_points_l.append(idx)
                decrease = False
        return detect_points_l

    def detect_sales_reboom_points(self, sales_thre = 5000):
        for genid in self.rst_d:
            if 'market_share_changes' in self.rst_d[genid] and np.sum(
                    self.rst_d[genid]['sales']) > sales_thre:
                self.rst_d[genid]['reboom'] = self.detect_repump(self.rst_d[genid]['market_share_changes'])

    def output_ms_file(self):
        with open('ms.csv', 'w') as f_out:
            f_out.write('Genmodel_ID,Genmodel,Automaker,' + ','.join([str(val) for val in range(2006, 2020)]) + '\n')

            for gid in self.rst_d:
                if 'market_shares' in self.rst_d[gid]:
                    f_out.write(','.join(
                        [gid, self.rst_d[gid]['Genmodel'], self.rst_d[gid]['Automaker']]) + ',')
                    f_out.write(','.join([str(val) for val in self.rst_d[gid]['market_shares']]))
                    f_out.write('\n')


    def assign_market_status(self):
        for gid in self.rst_d:
            if 'sales' not in self.rst_d[gid]: continue
            self.rst_d[gid]['market_status'] = []

            started = False
            for idx, val in enumerate(list(self.rst_d[gid]['sales'])):
                if val == 0:
                    t_label = 'B4' if started is False else 'End'
                else:
                    t_label = 'Alive'
                    started = True

                self.rst_d[gid]['market_status'].append(t_label)

    def read_aesthetic_predicitons(self, rating_fpa):
        rating_d = np.load(rating_fpa, allow_pickle=True).item()

        new_rating_d = {}
        for fpa, rating in zip(rating_d['img_na'], rating_d['Pred']):
            maker, model, year = fpa.split('/')[-1].split('$$')[:3]
            gid = fpa.split('/')[-1].split('$$')[4]
            if year not in new_rating_d:
                new_rating_d[year] = {}
            if gid not in new_rating_d[year]:
                new_rating_d[year][gid] = []
            new_rating_d[year][gid].append(rating)

        for year in new_rating_d:
            for gid in new_rating_d[year]:
                if 'aes' not in self.rst_d[gid]: self.rst_d[gid]['aes'] = {}

                self.rst_d[gid]['aes'][year] = [np.mean(new_rating_d[year][gid]), len(new_rating_d[year][gid])]

    def compute_seg_market_aes(self):
        self.seg_market_aes_d = {}

        for gid in self.rst_d:
            if 'aes' not in self.rst_d[gid]: continue

            for year in self.rst_d[gid]['aes']:
                if not 2018 > int(year) > 2006: continue

                b_type = self.rst_d[gid]['bodytype']
                if b_type not in self.seg_market_aes_d:
                    self.seg_market_aes_d[b_type] = {}
                if year not in self.seg_market_aes_d[b_type]:
                    self.seg_market_aes_d[b_type][year] = []
                self.seg_market_aes_d[b_type][year].append(self.rst_d[gid]['aes'][year][0])

    def compute_aes_rank(self):
        for gid in self.rst_d:
            if 'aes' not in self.rst_d[gid] or 'bodytype' not in self.rst_d[gid]: continue
            if self.rst_d[gid]['bodytype'] not in self.seg_market_aes_d: continue
            self.rst_d[gid]['aes_rank'] = {}

            for year in self.rst_d[gid]['aes']:
                if year not in self.seg_market_aes_d[self.rst_d[gid]['bodytype']]: continue
                self.rst_d[gid]['aes_rank'][year] = stats.percentileofscore(self.seg_market_aes_d[self.rst_d[gid]['bodytype']][year],
                                                          self.rst_d[gid]['aes'][year][0])

    def to_file_bodytype_dicts(self):
        body_d1 = {}
        body_d2 = {}

        for gid in self.rst_d:
            if 'bodytype' in self.rst_d[gid]:
                body_d1[gid] = self.rst_d[gid]['bodytype']
                t_key = (self.rst_d[gid]['Automaker'], self.rst_d[gid]['Genmodel'])
                body_d2[t_key] = self.rst_d[gid]['bodytype']

        return {'gid':body_d1, 'gmodel':body_d2}

# def get_model_avg_ratings(t_npa, img_d, tar_peri=(2013,2017)):
#     cont_d = {}
#     for row in t_npa:
#         year=int(row[0].split('$$')[2])
#         if year > tar_peri[1] or year < tar_peri[0] or row[0] not in img_d: continue
#         t_key = img_d[row[0]]
#         if t_key not in cont_d:
#             cont_d[t_key] = []
#         cont_d[t_key].append(float(row[1]))
#
#     for key in cont_d:
#         cont_d[key] = np.mean(cont_d[key])
#     return cont_d


# basic_tab_fpa = r'C:\Users\zernm\Desktop\My_codes\Facelift\evaluator\ws_dstore\shared_basic_table.csv'
# sales_table_pa = r'C:\Users\zernm\Desktop\My_codes\Facelift\evaluator\ws_dstore\shared_sales.csv'
# price_tab_fpa = r'C:\Users\zernm\Desktop\My_codes\Facelift\evaluator\ws_dstore\shared_price_table.csv'
# ad_tab_fpa = r'C:\Users\zernm\Desktop\My_codes\Facelift\evaluator\ws_dstore\shared_adv_table.csv'
# image_tab_fpa = r'C:\Users\zernm\Desktop\My_codes\Facelift\evaluator\ws_dstore\Image_table.csv'
#
# img_d = read_img_tab(image_tab_fpa)
# pred_rst_npa = np.load(r'C:\Users\zernm\Desktop\My_codes\Facelift\evaluator\ws_dstore\eval_on_exist_70.npy',
#                      allow_pickle=True)
# aes_pred_d = get_model_avg_ratings(pred_rst_npa, img_d, tar_peri=(2007, 2017))
#
# dvm_combiner = DVMCombiner(basic_tab_fpa)
# dvm_combiner.add_sale_table(sales_table_pa)
# dvm_combiner.add_price_tab(price_tab_fpa)
# dvm_combiner.add_bodytype(ad_tab_fpa)
# dvm_combiner.add_aesthetic(aes_pred_d)
# dvm_combiner.cal_market_shares()
# dvm_combiner.cal_market_share_changes()
# for key in dvm_combiner.rst_d:
#     if 'market_shares' in dvm_combiner.rst_d[key]:
#         print(key, dvm_combiner.rst_d[key]['market_share_changes'])
#
# print()