from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class MetalPriceDataset(Dataset):
    def __init__(self, data_type='train', pred_day=1, seq_len=1, data_scaler=None):
        super(MetalPriceDataset, self).__init__()
        self.data_type = data_type
        self.pred_day = pred_day
        self.data_scaler = data_scaler

        data_df, label_df = self.__get_pd_data(data_type)
        # 若当前为测试集, 还需要添加训练集中后seq_len-1行数据, 因此还需读取训练集
        if data_type == 'test':
            data_df_train, label_df_train = self.__get_pd_data(data_type='train')
            data_df = pd.concat([data_df_train, data_df], axis=0, sort=True)
            label_df = pd.concat([label_df_train, label_df], axis=0, sort=True)

        # 划分数据集
        if data_type == 'train':
            data_df = data_df['2004-01-01':'2016-12-31']
            label_df = label_df['2004-01-01':'2016-12-31']
        else:
            # 训练集和验证集分别加上前seq-1天的数据
            threshold_data = '2017-01-01' if data_type == 'valid' else '2018-01-01'
            data_len = len(data_df[threshold_data:]) + seq_len - 1
            data_df = data_df[-data_len:]
            label_df = label_df[-data_len:]

        self.data = data_df.values
        self.label = label_df.values

        # 数据归一化
        if self.data_scaler is not None:
            if data_type == 'train':
                self.data = self.data_scaler.fit_transform(self.data)
            else:
                self.data = self.data_scaler.transform(self.data)

        # 将data改为(n, seq_len, features_num)维度的数据
        # self.data行数为n+seq-1, 因此len(self.data) - seq_len + 1 = n
        t = np.empty([len(self.data) - seq_len + 1, seq_len, self.data.shape[-1]])
        t_label = np.empty([len(self.data) - seq_len + 1, seq_len, self.label.shape[-1]])
        for i in range(len(t)):
            t[i] = self.data[i: i + seq_len]
            t_label[i] = self.label[i: i + seq_len]
        self.data = t.astype(np.float32)
        self.label = t_label.astype(np.float32)

    def __get_pd_data(self, data_type):
        dir = 'data/Validation/Validation_data' if data_type == 'test' else 'data/Train/Train_data'
        dataset_name = 'validation' if data_type == 'test' else 'train'
        metal_list = ['Aluminium', 'Copper', 'Lead', 'Nickel', 'Tin', 'Zinc']

        # 读取LME数据
        lme_data = self.__get_lme_data(dir, dataset_name, metal_list)
        # 读取COMEX数据
        comex_data = self.__get_comex_data(dir, dataset_name)
        # 读取Indices数据
        indices_data = self.__get_indices_data(dir, dataset_name)

        # 拼接并填充
        data_df = pd.concat([lme_data, comex_data, indices_data], axis=1, sort=True)
        data_df.fillna(method='ffill', inplace=True)
        data_df.fillna(value=0, inplace=True)

        # 读取label
        label_df = self.__get_label(dir, dataset_name, metal_list, str(self.pred_day) + 'd')

        # 删除在训练数据中存在但不在label中的日期
        data_df = data_df[data_df.index.isin(label_df.index)]

        return data_df, label_df

    def __get_lme_data(self, dir, dataset_name, metal_list):
        lme_data = pd.DataFrame()
        for m in metal_list:
            oi = pd.read_csv('{}/LME{}_OI_{}.csv'.format(dir, m, dataset_name), index_col=0, usecols=[1, 2])
            tm = pd.read_csv('{}/LME{}3M_{}.csv'.format(dir, m, dataset_name), index_col=0, usecols=range(1, 7))
            # 当日开盘价与当天最高价之差
            tm['1'] = tm['Open.Price'] - tm['High.Price']
            # 当日开盘价与当天最低价之差
            tm['2'] = tm['Open.Price'] - tm['Low.Price']
            # 当日最高价与当日最低价的差
            tm['3'] = tm['High.Price'] - tm['Low.Price']
            # 当日最高价与昨日收盘价的差
            tm['4'] = tm['High.Price'] - tm['Close.Price'].shift(1)
            # 当日最低价与昨日收盘价的差
            tm['5'] = tm['Low.Price'] - tm['Close.Price'].shift(1)
            # 今日最高价与昨日最高价之差
            tm['6'] = tm['High.Price'] - tm['High.Price'].shift(1)
            # 昨日最低价与今日最低价之差
            tm['7'] = tm['Low.Price'].shift(1) - tm['Low.Price']
            x = pd.concat([oi, tm], axis=1, sort=True)
            x.columns = [m + '_' + col for col in x.columns]
            lme_data = pd.concat([lme_data, x], axis=1)
        return lme_data

    def __get_comex_data(self, dir, dataset_name):
        comex_metal = ['Copper', 'Gold', 'Platinum', 'Silver']
        comex_data = pd.DataFrame()
        for m in comex_metal:
            if m == 'Platinum':
                usecols = [1, 5]
            elif m == 'Silver':
                usecols = [1, 5, 6, 7]
            else:
                usecols = range(1, 8)
            x = pd.read_csv('{}/COMEX_{}_{}.csv'.format(dir, m, dataset_name), index_col=0, usecols=usecols)
            x.columns = ['COMEX_{}_{}'.format(m, col) for col in x.columns]
            comex_data = pd.concat([comex_data, x], axis=1)
        return comex_data

    def __get_indices_data(self, dir, dataset_name):
        indices_names = ['NKY Index', 'SHSZ300 Index', 'SPX Index', 'SX5E Index', 'UKX Index', 'VIX Index']
        indices_data = pd.DataFrame()
        for i in indices_names:
            x = pd.read_csv('{}/Indices_{}_{}.csv'.format(dir, i, dataset_name), index_col=0, usecols=[1, 2])
            indices_data = pd.concat([indices_data, x], axis=1, sort=True)
        return indices_data

    def __get_label(self, dir, dataset_name, metal_list, day):
        label = pd.DataFrame()
        if dataset_name == 'train':
            for m in metal_list:
                t = pd.read_csv('{}/Label_LME{}_{}_{}.csv'.format(dir, m, dataset_name, day),
                                index_col=0, usecols=[1, 2], header=0, names=[None, m + '_' + day])
                label = pd.concat([label, t], axis=1)
        else:
            x = pd.read_csv('data/validation_label_file.csv')
            x = pd.concat([x, x['id'].str.split('-', expand=True, n=3)], axis=1)
            for m in metal_list:
                t = x[(x[0] == 'LME' + m) & (x[2] == day)][['label', 3]]
                t.columns = [m + '_' + day, 'Index']
                t.set_index('Index', inplace=True)
                label = pd.concat([label, t], axis=1)
        return label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]

