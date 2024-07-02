# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from evaluation.eval_route import Metric
import torch.autograd as autograd


# import tensorflow as tf
def changeGraphOutput_etpa():
    g2r_input = ws + f'/data1/g2r_result.pkl'
    graph = np.load(g2r_input, allow_pickle=True)
    # Convert tensor values to NumPy ndarrays and store as a tuple
    ndarray_tuple = tuple(np.array(tensor) for tensor in graph.values())

    print("Tuple of NumPy ndarrays:")
    print(ndarray_tuple)


def get_workspace_etpa():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file


ws = get_workspace_etpa()


def batch_file_name_etpa(file_dir, suffix='.train'):
    """
    read all the paths with the same suffix under a file
    :param file_dir:
    :param suffix:
    :return:
    """
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                L.append(os.path.join(root, file))
    return L


def cmd_lst_etpa(cmd_lst=[]):
    """
    excute cmd list
    """
    for l in cmd_lst:
        cmd(l)


def cmd_etpa(cmd_):
    """
    excute and print one cmd
    """
    try:
        os.system(cmd_)
        print(cmd_)
    except:
        print(cmd_ + "failed")


def whether_stop_etpa(metric_lst=[], n=5, mode='maximize'):
    """
    For fast parameter search, judge wether to stop the training process according to metric score
    n: Stop training for n consecutive times without rising
    mode: maximize / minimize
    """
    if len(metric_lst) < 10: return False  # at least have 10 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx, v in enumerate(metric_lst):
        if v == max_v: max_idx = idx
    return max_idx < len(metric_lst) - n


class EarlyStop_ETPA:
    """
    For training process, early stop strategy
    """

    def __init__(self, mode='maximize', patience=5):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False

    def append(self, x):
        self.metric_lst.append(x)
        self.stop_flag = whether_stop_etpa(self.metric_lst, self.patience, self.mode)
        return self.stop_flag


def vis_distribution_etpa(data_lst=[], n=None, sections=None,
                          step=None, x_label='x', y_label='y', fout=''):
    """
    :param data_lst:
    :param n:   cut the x axis into n pieces
    :param sections: cut the x axis according to a list
    :param step:  cut the x axis in piceces with equal length (step)
    :param x_label:
    :param y_label:
    :return:
    """
    import platform
    if platform.system() != 'Windows': plt.switch_backend('agg')
    df = pd.DataFrame({x_label: data_lst, y_label: [1 for _ in data_lst]})
    assert (n is not None or sections is not None or step is not None), "Choose a n(int) or set a sections(list)"
    if n is not None:
        ys = pd.cut(df[x_label], n)
    if step is not None:
        min_, max_ = min(df[x_label]), max(df[x_label])
        sections = [min_ + i * step for i in range(int(max_ - min_))]
        ys = pd.cut(df[x_label], sections)
    if sections is not None:
        ys = pd.cut(df[x_label], sections)

    def get_stats(group):
        return {y_label: group.sum()}

    grouped = df[y_label].groupby(ys)
    df1 = grouped.apply(get_stats).unstack()
    plt.figure(figsize=(16, 12))
    # df1.plot(kind='pie', subplots=True)
    df1.plot(kind='bar')
    df1['ratio'] = df1.apply(lambda x: x[y_label] / sum(df1[y_label]), axis=1)
    if fout != '':
        dir_check(fout)
        plt.savefig(fout, figsize=())
    try:
        plt.show()
    except:
        pass
    return df1


def vis_distribution_auto_etpa(data_lst=[], fout=''):
    """
    :param fout:
    :param data_lst:
    :return:
    """
    import seaborn as sns
    sns.set_palette("hls")
    sns.displot(data_lst, kde_kws={"label": "pred"})
    # plt.title(f'avg:{avg(data_lst1)}-avg:{avg(data_lst2)}')
    if fout != '':
        plt.savefig(fout)
    try:
        plt.show()
    except:
        pass


def write_list_list_etpa(fp, list_, model="a", sep=","):
    dir = os.path.dirname(fp)
    if not os.path.exists(dir): os.makedirs(dir)
    f = open(fp, mode=model, encoding="utf-8")
    count = 0
    lines = []
    for line in list_:
        a_line = ""
        for l in line:
            l = str(l)
            a_line = a_line + l + sep
        a_line = a_line.rstrip(sep)
        lines.append(a_line + "\n")
        count = count + 1
        if count == 10000:
            f.writelines(lines)
            count = 0
            lines = []
    f.writelines(lines)
    f.close()


#  Both Dict_Merge Functions were same in ETPA/G2R
def dict_merge(dict_list=[]):
    """
    merge all the dict in the list
    """
    dict_ = {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_


import copy


def shuffle_label_etpa(label_index, label_len, top_k):
    new_index = copy.deepcopy(label_index)
    for x, l in zip(new_index, label_len):
        l = min(l, top_k)
        np.random.shuffle(x[:l])
    return new_index


def idx2order_etpa(idx):
    order = np.zeros_like(idx)
    for o, index in enumerate(idx):
        order[index] = o + 1 if index != -1 else 0
    return order


def batch_idx2order_etpa(index_np):
    return np.array(list(map(lambda idx: idx2order_etpa(idx), index_np)))


def dist_etpa(p1, p2, is_lat_first=True):
    """
    Calculate the distance between two points
    :param
        p1: the first point
        p2: the second point
        is_lat_first: is latidude first in the tuple
    :return:distance, unit: Meter
    """
    # print(p1, p2)
    from geopy.distance import geodesic
    if is_lat_first:
        d = geodesic(p1, p2).m
    else:
        d = geodesic((p1[1], p1[0]), (p2[1], p2[0]))
    return d


# For multi-thread
from multiprocessing import Pool


def multi_thread_work_etpa(parameter_queue, function_name, thread_number=5):
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return result


"""Utils for Training Deep Models"""
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def get_data_params_etpa(dataset='test_0-120_1-25'):
    raw_name_dict = {
        'test-b1': 'test',
        '22146-b1': 'gg_order_for_bjtu',
        'shanghai-b10': 'shanghai_10blocks',
        'shanghai-b1': 'gg_order_data_for_bjtu',
        'hangzhou-b10': 'hangzhou_10blocks',
        'shanghai-b50': 'shanghai_50blocks',
        'hangzhou-b1': 'gg_order_data_for_bjtu',
    }

    def get_days_etpa(fin):
        df = pd.read_csv(fin, sep=',', encoding='utf-8', header=None,
                         names=['日期', '运营区id', '城市', '快递员id', '接单时间', '预约时间1', '预约时间2',
                                '订单经度', '订单纬度',
                                '订单所属区块id', '区块类型id', '区块类型', '订单揽收时间',
                                '揽收最近时间', '揽收最近经度', '揽收最近纬度',
                                '揽收轨迹精度', '接单最近时间', '接单最近经度', '接单最近纬度',
                                '接单轨迹精度'])
        days = set(df['日期'].tolist())
        return len(days)

    params = {}
    params['data_file'] = ws + f'/data1/dataset/{dataset}/'
    if not 'artificial' in dataset:
        print("******* In Artifical Data *****")
        raw, time_range, len_range = dataset.split('_')
        raw_name = raw_name_dict.get(raw, raw)
        params['raw_path'] = ws + f'/data1/raw_data/{raw_name}.csv'
        params['temp_file'] = ws + f'/data1/temp/{raw_name}/'
        t_min, t_max = time_range.split('-')
        params['t_min'], params['t_max'] = int(t_min), int(t_max)

        l_min, l_max = len_range.split('-')
        params['l_min'], params['l_max'] = int(l_min), int(l_max)

        params['days'] = get_days_etpa(params['raw_path'])

    return params


def get_order_data_etpa(sort_mode, datatype):
    sort_data = sort_mode.split('_')[0].lower()
    if sort_data == 'none' or sort_data == 'true':
        file_path = ws + f'/data1/PnnOutput_{datatype}.pkl'
        print("*The file path is:", file_path)
        train_sort_idx, _, test_sort_idx, _ = np.load(file_path, allow_pickle=True)
        train_sort_idx = np.zeros_like(train_sort_idx)
        train_sort_pos = np.zeros_like(train_sort_idx)
        test_sort_idx = np.zeros_like(test_sort_idx)
        test_sort_pos = np.zeros_like(test_sort_idx)
        return train_sort_idx, train_sort_pos, test_sort_idx, test_sort_pos

    order_data = {
        'pnn': 'PnnOutput',
    }[sort_data]

    # file_path = ws + f'/data/{order_data}_{datatype}.pkl'

    #   ------------------------------ SRP Code --------------------------------------

    file_path = ws + f'/data1/route_result.pkl'
    print("*The file path is outside:", file_path)
    # Testing for Graph2Route file
    graph = np.load(file_path, allow_pickle=True)
    print(type(graph))
    # Convert tensor values to NumPy ndarrays and store as a tuple
    # Same Becuase we have 2 value from graph2Route and 4 in ranketpa
    # --------------------Before Masking ( Assign ranking to train and test )---------------------------------
    #  As the tensors inside dictionary were float -> first changed it to int values
    for key, tensor in graph.items():
        if tensor.dtype in [torch.float32, torch.float64]:
            # Use torch.floor() or torch.round() to convert float to int
            graph[key] = tensor.to(torch.int32)  # Convert to int32

    train_sort_idx, test_sort_idx = tuple(tensor for tensor in graph.values())

    # print("tensor", tensor.get_device(), type(tensor))

    # --------------------------- Masking ---------------------------------
    mask_value = torch.tensor(-1, dtype=torch.int32).to(device)
    # print("mask_value", mask_value.get_device(), type(mask_value))

    # Iterate through the dictionary and apply the mask
    for key, tensor in graph.items():
        mask = torch.eq(tensor, 26)  # Create a boolean mask where values are 26
        masked_tensor = torch.where(mask, torch.tensor(mask_value), tensor)
        graph[key] = masked_tensor
    # --------------------After Masking ( Assign masked ranking to train and test )---------------------------------

    train_sort_pos, test_sort_pos = tuple((tensor) for tensor in graph.values())
    # Original
    return train_sort_idx, train_sort_pos, test_sort_idx, test_sort_pos


class AverageMeter_etpa(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from evaluation.eval_etpa import eta_eval


class EtaMetric(object):
    # ETA metric
    def __init__(self):
        self.rmse = AverageMeter_etpa()
        self.mae = AverageMeter_etpa()
        self.mape = AverageMeter_etpa()
        self.acc10 = AverageMeter_etpa()
        self.acc20 = AverageMeter_etpa()
        self.acc30 = AverageMeter_etpa()

    def update(self, pred, label):
        for metric, record in zip(['rmse', 'mae', 'mape', 'acc@10', 'acc@20', 'acc@30'],
                                  [self.rmse, self.mae, self.mape, self.acc10, self.acc20, self.acc30]):
            value, n = eta_eval(pred, label.to(device), metric)
            record.update(value, n)

    def to_str(self):  # return a string for print
        _str_ = ''
        for metric, record in zip(['rmse', 'mae', 'mape', 'acc@10', 'acc@20', 'acc@30'],
                                  [self.rmse, self.mae, self.mape, self.acc10, self.acc20, self.acc30]):
            if metric in ['mape', 'acc@10', 'acc@20', 'acc@30']:
                _str_ += f'{metric}: {round(record.avg * 100, 2)}%\t'
            else:
                _str_ += f'{metric}: {round(record.avg, 2)}\t'
        return _str_

    def to_dict(self):  # reuturn a dict
        dict = {}
        for metric, record in zip(['rmse', 'mae', 'mape', 'acc@10', 'acc@20', 'acc@30'],
                                  [self.rmse, self.mae, self.mape, self.acc10, self.acc20, self.acc30]):
            dict[metric] = record.avg
        return dict


# Loss Function
def mask_loss(pred, label):
    # print("Prediction is:\n", pred)
    # print("Label is:\n", label)
    # print("preeed", pred.get_device(), type(pred))
    loss_func = nn.HuberLoss().to(pred.device)
    mask = label > 0
    label = label.masked_select(mask).to(device)
    pred = pred.masked_select(mask.to(device))
    loss = loss_func(pred, label)
    n = mask.sum().item()

    # print("mask", mask.get_device(), type(mask))
    # print("Prediction is:\n", pred)
    # print("Label is:\n", label)
    return loss, n


# the label information are put into the same x
class MyDataset(Dataset):
    def __init__(self, last_x, last_len, unpick_x, unpick_len,
                 label_idx, label_order, label_eta, pnn_idx, pnn_pos, start, end):
        super(MyDataset, self).__init__()
        self.len = end - start
        self.last_x = last_x[start:end]
        self.last_len = last_len[start:end]
        # self.global_x = global_x[start:end]
        self.unpick_x = unpick_x[start:end]
        self.unpick_len = unpick_len[start:end]
        self.label_idx = label_idx[start:end]
        self.label_eta = label_eta[start:end]
        self.label_order = label_order[start:end]
        self.pnn_idx = pnn_idx[start:end]
        self.pnn_pos = pnn_pos[start:end]

        # size dict
        self.size_dict = {
            'last_x_size': len(last_x[0][0]),
            'unpick_x_size': len(unpick_x[0][0]),
            # 'global_x_size': len(global_x[0])
        }

    def __len_etpa__(self):
        return self.len

    def __getitem__(self, index):
        return (self.last_x[index]).float(), \
            self.last_len[index], \
            (self.unpick_x[index]).float(), \
            torch.from_numpy(self.unpick_len[index]), \
            torch.from_numpy(self.label_idx[index]).int(), \
            torch.from_numpy(self.label_order[index]).int(), \
            (self.label_eta[index]), \
            (self.pnn_idx[index]), \
            (self.pnn_pos[index])

    def get_size_etpa(self):
        # input_size = len(self.last_x[0][0]) * 5  + len(self.global_x[0]) + len(self.unpick_x[0][0]) * max_len + 1 + max_len
        # change 25- > 27

        # input_size = len(self.global_x[0]) + len(self.unpick_x[0][0]) + 27 + 1
        input_size = len(self.unpick_x[0][0]) + 27 + 1
        sort_x_size = len(self.unpick_x[0][0])
        return input_size, sort_x_size

    # from abc import abstractmethod
    # @abstractmethod
    def get_input_size_etpa(self):  # redefine this function for different method
        # change 25- > 27

        # return len(self.global_x[0]) + len(self.unpick_x[0][0]) + 27 + 1
        return len(self.unpick_x[0][0]) + 27 + 1

    def size_etpa(self):
        self.size_dict['input_size'] = self.get_input_size_etpa()
        return self.size_dict


# ----------------------------------------------Added - SRP --------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)


def preprocess_food(V, V_ft, V_dt, V_dispatch_mask, cou, V_val, pad_value):
    # Get shape
    B, T, N = V_dispatch_mask.shape

    # Initialize the arrays for batch data
    last_x_idx = torch.full((B, T, 5), pad_value, dtype=torch.float).to(device)
    last_x_mask = torch.zeros((B, T, N), dtype=torch.float).to(device)
    last_len = torch.zeros((B, T), dtype=torch.float).to(device)
    unpicked_x_idx = torch.full((B, T, N), pad_value, dtype=torch.float).to(device)
    unpicked_x_mask = torch.zeros((B, T, N), dtype=torch.float).to(device)
    unpicked_len = torch.zeros((B, T), dtype=torch.float).to(device)

    #  A
    eta_np = torch.zeros((B, N), dtype=torch.float).to(device)  # torch.Size([128, 27])
    order_np = torch.zeros((B * T, N), dtype=torch.float).to(device)
    index_np = torch.zeros((B * T, N), dtype=torch.float).to(device)

    # print("device is", device)

    # Iterate through the number of courier
    for b in range(B):

        # A
        for i, value in enumerate(V_ft[b]):
            result = 0
            if value != 0:
                result = value - V_ft[b][0]
                eta_np[b][i] = result

        # Compute the array of unique and sorted dispatch time
        dispatch_times = np.sort(np.unique(V_dt[b][np.nonzero(V_dt[b])]))
        # Iterate through the dispatch times
        for t, dt in enumerate(dispatch_times):
            # Define temp arrays
            temp_unpicked_id = []
            temp_picked_id, temp_picked_time = [], []
            temp_last_x_mask = np.zeros(N)

            # Compute the active nodes at time step t
            active_nodes = V_ft[b] * V_dispatch_mask[b][t]

            # Iterate through the active nodes
            for idx, ft in enumerate(active_nodes):
                # If the node has been picked
                if (t == len(dispatch_times) - 1 and ft > dt) or (ft < dt and ft != 0):
                    # Append the id and ft to the temp array
                    temp_picked_id.append(idx)
                    temp_picked_time.append(ft)

                # If the node is dispatched but not picked
                elif ft != 0:
                    # Append the id to the temp array
                    temp_unpicked_id.append(idx)
                    unpicked_x_mask[b, t, idx] = 1

            # Sort the picked id by descending order and take the top 5 (for last_x)
            sorted_picked_idx = np.argsort(temp_picked_time)[::-1][:5]

            # Convert the list into np array and add padding
            last_x_idx_t = np.pad(np.array(temp_picked_id, dtype=int)[sorted_picked_idx],
                                  (0, 5 - len(sorted_picked_idx)), 'constant', constant_values=(pad_value))
            unpicked_x_idx_t = np.pad(temp_unpicked_id, (0, N - len(temp_unpicked_id)), 'constant',
                                      constant_values=(pad_value))

            # Create the last_x_mask
            temp_last_x_mask[last_x_idx_t[(last_x_idx_t != pad_value)]] = 1

            # Insert the temp data to the batch tensor
            last_x_idx[b, t, :] = torch.from_numpy(last_x_idx_t).to(device)
            unpicked_x_idx[b, t, :] = torch.from_numpy(unpicked_x_idx_t).to(device)
            last_x_mask[b, t, :] = torch.from_numpy(temp_last_x_mask).to(device)

            # Count the nonzero values of the mask tensors as the len
            unpicked_len[b, t] = torch.count_nonzero(unpicked_x_mask[b, t, :]).to(device)
            last_len[b, t] = torch.count_nonzero(last_x_mask[b, t, :]).to(device)

    # Convert input to tensors
    V = torch.FloatTensor(V).to(device)
    V_ft = torch.FloatTensor(V_ft).to(device)
    V_dt = torch.FloatTensor(V_dt).to(device)
    cou = torch.FloatTensor(cou).to(device)
    V_val = V_val.to(device)

    # Assemble the features (concat V_val and cou data)
    unpicked_feats = torch.cat([V_val, cou.unsqueeze(1).unsqueeze(1).repeat((1, T, N, 1))], axis=3).to(device)
    picked_feats = torch.cat([V_ft.unsqueeze(2).unsqueeze(1).repeat((1, T, 1, 1)), unpicked_feats], axis=3).to(device)

    # Compute the unpicked_x and last_x by multiply it by the mask
    unpicked_x = unpicked_feats * unpicked_x_mask.unsqueeze(3).repeat(1, 1, 1, unpicked_feats.shape[3])
    last_x = picked_feats * last_x_mask.unsqueeze(3).repeat(1, 1, 1, picked_feats.shape[3])

    # Reshape output
    unpicked_x = unpicked_x.reshape((B * T, N, unpicked_feats.shape[3]))
    unpicked_len = unpicked_len.reshape((B * T))
    last_x = last_x.reshape((B * T, N, picked_feats.shape[3]))
    last_len = last_len.reshape((B * T))

    eta_np = (eta_np.unsqueeze(1).repeat((1, 12, 1)) * unpicked_x_mask).reshape(B * T, N).detach().cpu().numpy()

    # A
    # Mask zero values with np.inf
    temp_masked_array = np.where(eta_np == 0, np.inf, eta_np)  # Shape is (128,27)
    non_inf_mask = ~np.isinf(temp_masked_array)

    # temp_count_array -> to store the number of non-zero values
    temp_count_array = np.sum(non_inf_mask == True, axis=1)
    # Get the indices that would sort the array while ignoring masked values
    sorted_indices_array = np.argsort(temp_masked_array)
    masks = [np.arange(len(row)) < count for row, count in zip(sorted_indices_array, temp_count_array)]

    # Apply the masks to the array
    order_np = np.where(np.stack(masks), sorted_indices_array, 0)
    index_np = order_np - 1

    # Reshape for joint training
    unpicked_x = unpicked_x.reshape((B, T, N, -1))
    unpicked_len = unpicked_len.reshape((B, T))
    last_x = last_x.reshape((B, T, N, -1))
    last_len = last_len.reshape((B, T))
    eta_np = eta_np.reshape((B, T, N))
    order_np = order_np.reshape((B, T, N))
    index_np = index_np.reshape((B, T, N))

    return unpicked_x.detach().cpu().numpy(), unpicked_len.detach().cpu().numpy(), last_x.detach().cpu().numpy(), last_len.detach().cpu().numpy(), eta_np, order_np, index_np


# --------------------------------------------Preprocess Food Dataset------------------------------------------------------


def get_train_val_etpa(data_path, node_features_file, sort_idx, sort_pos, val_split=0.25, ModelDataset=MyDataset):
    #  External Data
    # -------------------------------Change --------------------------------------------

    # Load the data files
    data = np.load(data_path, allow_pickle=True).item()
    node_features = np.load(node_features_file, allow_pickle=True)
    V = data['V']
    V_ft = data['V_ft']
    V_dt = data['V_dt']
    V_dispatch_mask = data['V_dispatch_mask']
    cou = data['cou']
    V_val = node_features['V_val']
    # Hyperparameter
    pad_value = 26
    unpick_x, unpick_len, last_x, last_len, eta_np, order_np, index_np = preprocess_food(V, V_ft, V_dt, V_dispatch_mask,
                                                                                         cou, V_val, pad_value)
    # ------------------------------------------------------------------------------------------

    length = len(unpick_len)
    val_flag = int(length * (1 - val_split))

    train_set = ModelDataset(last_x, last_len, unpick_x, unpick_len, index_np, order_np, eta_np, sort_idx, sort_pos, 0,
                             val_flag)
    val_set = ModelDataset(last_x, last_len, unpick_x, unpick_len, index_np, order_np, eta_np, sort_idx, sort_pos,
                           val_flag, length)

    return train_set, val_set


import time
import nni, math


def get_model_function_etpa():
    import ranketpa.RankPETA as RankPETA

    # model_dict = {
    #     'RankPETA': (RankPETA.MyModel, RankPETA.save2file),
    # }
    # Model, Save2File = model_dict[model]
    return RankPETA.MyModel, RankPETA.save2file


from pprint import pprint
import argparse

if __name__ == '__main__':
    pass

# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pickle


def get_workspace_g2r():
    """
    get the workspace path, i.e., the root directory of the project
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file


ws = get_workspace_g2r()


def dir_check(path):
    """
    check weather the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)


def whether_stop(metric_lst=[], n=2, mode='maximize'):
    """
    For fast parameter search, judge wether to stop the training process according to metric score
    n: Stop training for n consecutive times without rising
    mode: maximize / minimize
    """
    if len(metric_lst) < 1: return False  # at least have 1 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx, v in enumerate(metric_lst):
        if v == max_v: max_idx = idx
    return max_idx < len(metric_lst) - n


from multiprocessing import Pool


def multi_thread_work_g2r(parameter_queue, function_name, thread_number=5):
    """
    For parallelization
    """
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return result


class EarlyStop_G2R():
    """
    For training process, early stop strategy
    """

    def __init__(self, mode='maximize', patience=1):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1  # the best epoch
        self.is_best_change = False  # whether the best change compare to the last epoch

    def append(self, x):
        """
        append a value, then update corresponding variables
        """
        self.metric_lst.append(x)
        # update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        # update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize' else self.metric_lst.index(
            min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch  # update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        """
        return the best metric
        """
        if len(self.metric_lst) == 0:
            return -1
        else:
            return self.metric_lst[self.best_epoch]


def batch_file_name_g2r(file_dir, suffix='.train'):
    """
    Find all files whose suffix is [suffix] in given directory [file_dir]
    """
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                L.append(os.path.join(root, file))
    return L


# Change Get Dataset
def get_dataset_path(params={}):
    """
    get file path of train, validate and test dataset
    """
    if params['modelRoute'] == 'graph2route_pd':
        dataset = 'food_pd'
    else:
        dataset = 'logistics'
    params['dataset'] = dataset
    file = ws + f'/data/dataset/{dataset}'
    train_path = file + f'/train_full.npy'
    val_path = file + f'/val_full.npy'
    test_path = file + f'/test_full.npy'
    # for node features
    node_features_train = file + f'/node_features_train_full.npy'
    node_features_val = file + f'/node_features_val_full.npy'
    node_features_test = file + f'/node_features_test_full.npy'

    return train_path, val_path, test_path, node_features_train, node_features_val, node_features_test


def write_list_list_g2r(fp, list_, model="a", sep=","):
    dir = os.path.dirname(fp)
    if not os.path.exists(dir): os.makedirs(dir)
    f = open(fp, mode=model, encoding="utf-8")
    count = 0
    lines = []
    for line in list_:
        a_line = ""
        for l in line:
            l = str(l)
            a_line = a_line + l + sep
        a_line = a_line.rstrip(sep)
        lines.append(a_line + "\n")
        count = count + 1
        if count == 10000:
            f.writelines(lines)
            count = 0
            lines = []
    f.writelines(lines)
    f.close()


# This function - Only in G2R
def save2file_meta(params, file_name, head):
    """
    functions for saving results
    """

    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t

    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()
    with open(file_name, "a", newline='\n') as file:  # linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)


# ----- Training Utils----------
import argparse
import random, torch
from torch.optim import Adam
from pprint import pprint
from torch.utils.data import DataLoader


def get_common_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--is_test', type=bool, default=False, help='test the code')
    # dataset
    parser.add_argument('--datatype', default='order', type=str, help='datatype')  # ETPA
    parser.add_argument('--worker_emb_dim', type=int, default=10, help='embedding dimension of a worker')

    parser.add_argument('--min_task_num', type=int, default=0, help='minimal number of task')
    parser.add_argument('--max_task_num', type=int, default=25, help='maxmal number of task')
    parser.add_argument('--dataset', default='food_pd', type=str, help='food_pd or logistics')
    parser.add_argument('--pad_value', type=int, default=26, help='food servce, max node num is 25 ')

    ## common settings for deep models
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--num_epoch', type=int, default=1000, help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=1999, metavar='S', help='random seed (default: 6)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop at')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 4)')
    parser.add_argument('--is_eval', type=str, default=False, help='True means load existing model')

    parser.add_argument('--num_worker_pd', type=int, default=920, help='number of workers in food delivery dataset')
    parser.add_argument('--num_worker_logistics', type=int, default=2346, help='number of workers in logistics dataset')

    parser.add_argument('--spatial_encoder', type=str, default='gcn', help='type of spatial encoder')
    parser.add_argument('--temporal_encoder', type=str, default='gru', help='type of temporal encoder')

    # common settings for graph2route model
    parser.add_argument('--node_fea_dim', type=int, default=8, help='dimension of node input feature')
    parser.add_argument('--edge_fea_dim', type=int, default=5, help='dimension of edge input feature')
    parser.add_argument('--hidden_size', type=int, default=8)
    parser.add_argument('--gcn_num_layers', type=int, default=2)
    parser.add_argument('--k_nearest_neighbors', type=str, default='n-1')
    parser.add_argument('--k_min_nodes', type=int, default=3)
    # settings for evaluation
    parser.add_argument('--eval_start', type=int, default=1)
    parser.add_argument('--eval_end_1', type=int, default=11)
    parser.add_argument('--eval_end_2', type=int, default=25)
    #  These three are from etpa common_params
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--sort_mode', type=str, default="pnn_idx", metavar='N', help='pnn_pos/pnn_idx')
    parser.add_argument('--train_mode', type=str, default="sort", metavar='N', help='')
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N', help='')
    parser.add_argument('--num_block', type=int, default=2, metavar='N', help='')
    parser.add_argument('--num_head', type=int, default=2, metavar='N', help='')

    parser.add_argument('--modelRoute', type=str, default="graph2route_pd", metavar='N', help='')
    parser.add_argument('--modelArrivalTime', type=str, default="RankPETA", metavar='N', help='')
    parser.add_argument('--alpha_loss', type=float, default=0.5, metavar='N', help='')
    parser.add_argument('--scheduled_alpha', type=bool, default=True, metavar='N', help='')
    return parser


def filter_data_g2r(data_dict={}, len_key='node_len', min_len=0, max_len=20):
    '''
    filter data, For dataset
    '''
    new_dic = {}
    keep_idx = [idx for idx, l in enumerate(data_dict[len_key]) if l >= min_len and l <= max_len]
    for k, v in data_dict.items():
        new_dic[k] = [data for idx, data in enumerate(data_dict[k]) if idx in keep_idx]
    return new_dic


def to_device_g2r(batch, device):
    batch = [x.to(device) for x in batch]
    return batch


import nni, time


def train_val_test_g2r(train_loader, val_loader, test_loader, modelRoute, device, process_batch, test_model, params,
                       save2FileRoute, save2FileArrivalTime, modelArrivalTime):
    modelRoute.to(device)
    modelArrivalTime.to(device)

    min_val_rmse = 9999
    train_loss = AverageMeter_etpa()
    optimizer = Adam([{'params': modelRoute.parameters()}, {'params': modelArrivalTime.parameters()}], lr=params['lr'],
                     weight_decay=params['wd'])

    # Early Stop Implementation
    early_stop_g2r = EarlyStop_G2R(mode='maximize', patience=params['early_stop'])
    early_stop_etpa = EarlyStop_ETPA(mode='maximize', patience=params['early_stop'])

    # Evaluation Metric for ETPA
    val_result_etpa, test_result_etpa = EtaMetric(), EtaMetric()

    modelRoute_name = modelRoute.model_file_name() + f'{time.time()}'
    modelArrivalTime_name = modelArrivalTime.model_file_name()

    modelRoute_path = ws + f'/data/dataset/{params["dataset"]}/sort_model/{modelRoute_name}'
    modelArrivelTime_path = ws + f'/data/eta_model/{modelArrivalTime_name}'

    dir_check(modelRoute_path)
    dir_check(modelArrivelTime_path)

    torch.cuda.amp.autocast(enabled=False)

    train_loss_output = np.array([])
    val_loss_output = np.array([])

    # Training
    for epoch in range(params['num_epoch']):
        if early_stop_g2r.stop_flag and early_stop_etpa.stop_flag: break
        postfix = {"epoch": epoch, "loss": 0.0, "current_loss": 0.0}  # Combine Parameters , Epochs, loss, currentlos
        # for alpha in alpha_search_values:
        #     print(f"Training for alpha = {alpha}")
        # Initialize tensor for total pointers
        total_pred_pointers = torch.tensor([]).to(device)
        total_node_features = torch.tensor([]).to(device)
        total_index = torch.tensor([], dtype=torch.int64).to(device)
        train_loss_batch = np.array([])
        print(f"*** Epoch: {epoch} ***")

        # Define the scheduled alpha
        scheduled_alpha = 0.9
        if epoch >= 20:
            scheduled_alpha = 0.1
        elif epoch >= 15:
            scheduled_alpha = 0.3
        elif epoch >= 10:
            scheduled_alpha = 0.5
        elif epoch >= 5:
            scheduled_alpha = 0.7

        with tqdm(train_loader, total=len(train_loader), postfix=postfix) as t:
            ave_loss_g2r = None
            modelRoute.train()
            modelArrivalTime.train()
            eval_batch = len(train_loader) // 10  # evaluate the model  about 10 times for each epoch
            if eval_batch == 0: eval_batch = len(train_loader) - 1

            for i, batch in enumerate(t):
                # Compute the route prediction using the Graph2Route model
                pred, loss_g2r, b_V_val_unmasked, index = process_batch(batch, modelRoute, device, params['pad_value'])

                total_pred_pointers = torch.cat((total_pred_pointers, pred), 0)
                total_node_features = torch.cat((total_node_features, b_V_val_unmasked), 0)
                total_index = torch.cat((total_index, index), 0)

                if ave_loss_g2r is None:
                    ave_loss_g2r = loss_g2r.item()
                else:
                    ave_loss_g2r = ave_loss_g2r * i / (i + 1) + loss_g2r.item() / (i + 1)
                postfix["loss"] = ave_loss_g2r  # Check postfix[loss]
                postfix["current_loss"] = loss_g2r.item()
                t.set_postfix(**postfix)

                torch.autograd.set_detect_anomaly(True)

                #  Code for pnn,
                # --------------------Before Masking ( Assign ranking to train and test )---------------------------------
                #  As the tensors inside dictionary were float -> first changed it to int values

                pred = pred.to(torch.int32)  # Convert to int32
                # --------------------------- Masking ---------------------------------
                mask_value = torch.tensor(-1, dtype=torch.int32).to(device)
                mask = torch.eq(pred, 26)  # Create a boolean mask where values are 26
                masked_tensor = torch.where(mask, mask_value, pred)
                # --------------------After Masking ( Assign masked ranking to train and test )---------------------------------
                sort_idx = pred
                sort_pos = masked_tensor
                # # =============================Changes For ETPA==================================================
                # Changed data to batch because we are enumerating over t in g2r and getting  i and batch
                _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, cou, _, last_x, last_len, unpick_x, unpick_len, label_eta, label_order, label_idx, _ = zip(
                    *batch)
                #  change the data type of ETPA parameters tuples->arrays
                last_x = np.array(last_x)
                last_len = np.array(last_len)
                unpick_x = np.array(unpick_x)
                unpick_len = np.array(unpick_len)
                label_eta = np.array(label_eta)
                label_order = np.array(label_order)
                label_idx = np.array(label_idx)

                #  change the data type of ETPA parameters arrays->Tensors
                last_x = torch.FloatTensor(last_x)
                last_len = torch.FloatTensor(last_len)
                unpick_x = torch.FloatTensor(unpick_x)
                unpick_len = torch.FloatTensor(unpick_len)
                label_eta = torch.FloatTensor(label_eta)
                label_order = torch.LongTensor(label_order)
                label_idx = torch.LongTensor(label_idx)

                (input_idx, input_order) = (label_idx, label_order) if params['train_mode'] == 'true' else (
                sort_idx, sort_pos)

                # Reshape input to B*T ...
                _B, _T, _N = label_eta.shape
                last_x = last_x.reshape((_B * _T, _N, -1))
                last_len = last_len.reshape((_B * _T))
                unpick_x = unpick_x.reshape((_B * _T, _N, -1))
                unpick_len = unpick_len.reshape((_B * _T))
                label_eta = label_eta.reshape((_B * _T, _N))
                label_order = label_order.reshape((_B * _T, _N))
                label_idx = label_idx.reshape((_B * _T, _N))

                # Compute the arrival time estimation using the predicted route and additional raw features
                pred_eta, loss_etpa, n = modelArrivalTime(last_x, last_len, unpick_x, unpick_len, label_idx,
                                                          label_order, label_eta, input_idx, input_order)

                optimizer.zero_grad()

                # Compute the joint loss
                alpha = scheduled_alpha if params['scheduled_alpha'] is True else params['alpha_loss']
                joint_loss = alpha * loss_g2r + (1 - alpha) * (loss_etpa / 10)
                train_loss_batch = np.append(train_loss_batch, joint_loss.cpu().detach().numpy())

                if torch.isnan(joint_loss):
                    print(" Detected NaN loss. Skipping backward pass.")
                else:
                    joint_loss.backward()
                    optimizer.step()

                eta_mse, n = mask_loss(pred_eta, label_eta)
                train_loss.update(eta_mse.item(), n)
                """if i % eval_batch == 0:
                    print('Epoch {}: Train [{}/{} ({:.0f}%)]\tRMSE: {:.6f}'
                        .format(epoch, i * len(last_x), len(train_loader.dataset),
                                100. * i / len(train_loader), math.sqrt(train_loss.avg)))"""

            train_loss_output = np.append(train_loss_output, np.mean(train_loss_batch))

            if params['is_test']: break

            # # Define the file name of the output file (.pkl)
            # output_fname = f'route_result_{params["spatial_encoder"]}_{params["temporal_encoder"]}_{params["seed"]}.pkl'
            #
            # # Initialize the dictionary for the output
            # output_dict = {}
            #
            # # Check if file exists
            # if Path(output_fname).is_file():
            #     output_dict = np.load(output_fname, allow_pickle=True)
            #
            # # Append the data to the dictionary
            # output_dict[str(epoch)] = total_pred_pointers
            #
            # # Store the data to the file
            # with open(output_fname, 'wb') as df_file:
            #     pickle.dump(obj=output_dict, file=df_file)
            #
            # # Store node feature data into a file
            # output_dict_node = {}
            # output_dict_node['V_val'] = total_node_features
            # output_fname_node = 'node_features_train_.npy'
            # with open(output_fname_node, 'wb') as df_file:
            #     pickle.dump(obj=output_dict_node, file=df_file)
            #
            # # Store node feature data into a file
            # output_dict_index = {}
            # output_dict_index['index'] = total_index
            # output_fname_index = f'index_train_{params["spatial_encoder"]}_{params["temporal_encoder"]}_{params["seed"]}.npy'
            # with open(output_fname_index, 'wb') as df_file:
            #     pickle.dump(obj=output_dict_index, file=df_file)

            # Validation process
            val_result_g2r, pred_etpa, val_lable_eta, val_loss_epoch = test_model(modelRoute, val_loader, device, params['pad_value'],
                                                                  params, save2FileRoute, 'val', modelArrivalTime,
                                                                  save2FileArrivalTime)
            val_loss_output = np.append(val_loss_output, val_loss_epoch)
            val_result_etpa.update(pred_etpa, val_lable_eta)
            #  Early Stop - G2R And file Save
            print("-----------------G2R RESULT-------------------")
            print('\nval result:', val_result_g2r.to_str(), 'Best krc:', round(early_stop_g2r.best_metric(), 3),
                  '| Best epoch:', early_stop_g2r.best_epoch)
            is_best_change = early_stop_g2r.append(val_result_g2r.to_dict()['krc'])

            if is_best_change:
                # print('value:',val_result_g2r.to_dict()['krc'], early_stop_g2r.best_metric())
                torch.save(modelRoute.state_dict(), modelRoute_path)
                print('best model saved')
                # print('model path:', modelRoute_path)

            if params['is_test']:
                # print('model_path:', modelRoute_path)
                torch.save(modelRoute.state_dict(), modelRoute_path)
                print('best model saved !!!')
                break
            nni.report_intermediate_result(val_result_g2r.to_dict()['krc'])
            #  Early Stop ETPA and file Save
            print("-----------------ETPA RESULT-------------------")
            print(f'Epoch {epoch}: Validation\t' + val_result_etpa.to_str())
            if val_result_etpa.rmse.avg < min_val_rmse:
                min_val_rmse = val_result_etpa.rmse.avg
                val_mae = val_result_etpa.mae.avg
                final_result = val_result_etpa
                best_epoch = epoch
                torch.save(modelArrivalTime.state_dict(), modelArrivelTime_path)
            # print('Best epoch in validation: ' + str(best_epoch) + '\tbest RMSE: ' + str(min_val_rmse)+ '\tMAE: ' + str(val_mae))
            nni.report_intermediate_result(val_result_etpa.rmse.avg)
            early_stop_etpa.append(val_result_etpa.rmse.avg)
            if params['is_test']: break
            if 'mape: 100.0%' in val_result_etpa.to_str():
                print('mape=100.0%, the training process is stopped')
                early_stop_etpa.stop_flag = True

    try:
        print('loaded model path:', modelRoute_path)
        modelRoute.load_state_dict(torch.load(modelRoute_path))
        modelArrivalTime.load_state_dict(torch.load(modelArrivelTime_path))
        print('best model loaded !!!')
    except:
        print('load best model failed')

    # # Define the file name of the output file (.pkl)
    # output_fname = f'route_result_{params["spatial_encoder"]}_{params["temporal_encoder"]}_{params["seed"]}.pkl'
    #
    # # Initialize the dictionary for the output
    # temp_dict = {}
    # output_dict = {}
    #
    # # Check if file exists
    # if Path(output_fname).is_file():
    #     temp_dict = np.load(output_fname, allow_pickle=True)
    #
    # output_dict['pred_pointers_train'] = temp_dict[str(early_stop_g2r.best_epoch)]
    #
    # # Store the data to the file
    # with open(output_fname, 'wb') as df_file:
    #     pickle.dump(obj=output_dict, file=df_file)
    #
    # # Store the loss
    # loss_dict = {}
    # loss_dict['train'] = train_loss_output
    # loss_dict['val'] = val_loss_output
    # output_fname_loss = f'loss_joint_{params["spatial_encoder"]}_{params["temporal_encoder"]}_{params["seed"]}.npy'
    # with open(output_fname_loss, 'wb') as df_file:
    #     pickle.dump(obj=loss_dict, file=df_file)

    # Test process
    test_result, test_pred_etpa, test_lable_eta = test_model(modelRoute, test_loader, device, params['pad_value'],
                                                             params, save2FileRoute, 'test', modelArrivalTime,
                                                             save2FileArrivalTime)
    test_result_etpa.update(test_pred_etpa, test_lable_eta)
    print('\n-------------------------------------------------------------')
    print('Best epoch for G2R: ', early_stop_g2r.best_epoch)
    print('Best epoch for ETPA: ', best_epoch)

    # # Check this test_result type
    print(f'{params["modelRoute"]} Evaluation in test:', test_result.to_str())
    print(f'{params["modelArrivalTime"]} Evaluation in test:', test_result_etpa.to_str())

    # nni.report_final_result(test_result.to_dict()['krc'])
    return params, joint_loss.item(), dict_merge([test_result_etpa.to_dict(), {'train_time': params['spatial_encoder'], 'test_time': params['temporal_encoder'], 'stop_epoch': best_epoch}])


def evaluate_performance(alpha_values, loss_g2r, loss_etpa):
    best_alpha = None
    best_joint_loss = float('inf')  # Initialize with a large value

    for alpha in alpha_values:
        joint_loss = alpha * loss_g2r + (1 - alpha) * loss_etpa

        # Choose the alpha that minimizes the joint loss
        if joint_loss < best_joint_loss:
            best_alpha = alpha
            best_joint_loss = joint_loss

    return best_alpha, best_joint_loss


# Only G2R is using get-nonzeros_function
def get_nonzeros(pred_steps, label_steps, label_len, pred_len, pad_value):
    pred = []
    label = []
    label_len_list = []
    pred_len_list = []
    for i in range(pred_steps.size()[0]):
        # remove samples with no label
        if label_steps[i].min().item() != pad_value:
            label.append(label_steps[i].cpu().numpy().tolist())
            pred.append(pred_steps[i].cpu().numpy().tolist())
            label_len_list.append(label_len[i].cpu().numpy().tolist())
            pred_len_list.append(pred_len[i].cpu().numpy().tolist())
    return torch.LongTensor(pred), torch.LongTensor(label), \
        torch.LongTensor(label_len_list), torch.LongTensor(pred_len_list)


def get_model_function_g2r():
    import graph2route.graph2route_pd.model as graph2route_pd
    #import graph2route.graph2route_logistics.model as graph2route_logistics

    # model_dict = {
    #
    #     'graph2route_pd': (graph2route_pd.Graph2Route, graph2route_pd.save2file),
    #     #'graph2route_logistics': (graph2route_logistics.Graph2Route, graph2route_logistics.save2file),
    #
    # }
    # model, save2file = model_dict[model]
    return graph2route_pd.Graph2Route, graph2route_pd.save2file


#  Only one run function we have and it is from g2r
# it was run in g2r
def run(params, DATASET, PROCESS_BATCH, TEST_MODEL, collate_fn=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(params.get('seed', 2022))
    params['device'] = device
    params['train_path'], params['val_path'], params['test_path'], params['node_features_train'], params[
        'node_features_val'], params['node_features_test'] = get_dataset_path(params)
    pprint(params)  # print the parameters

    train_dataset = DATASET(mode='train', params=params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn,
                              drop_last=True)

    val_dataset = DATASET(mode='val', params=params)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn,
                            drop_last=True)

    test_dataset = DATASET(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn,
                             drop_last=True)

    # train, valid and test
    # net_models = ['graph2route_pd']
    # print("Model route is :",params['modelRoute'])
    # print("Model ETPA is :",params['modelArrivalTime'])

    # Get the Graph2Route model and initialize it
    modelRoute, save2FileRoute = get_model_function_g2r()
    modelRoute = modelRoute(params)

    # Get the RankETPA model and initialize it
    modelArrivalTime, save2FileArrivalTime = get_model_function_etpa()
    size_dict = train_dataset.size()
    print(';'.join([f'{k}:{v}' for k, v in size_dict.items()]))
    print(f"#train:{len(train_dataset)} | #val:{len(val_dataset)} | #test:{len(test_dataset)}")

    params = dict_merge([size_dict, params])
    modelArrivalTime = modelArrivalTime(params)

    # Start the train, val, and test process
    result_dict, loss, etpa_result = train_val_test_g2r(train_loader, val_loader, test_loader, modelRoute, device, PROCESS_BATCH,
                                     TEST_MODEL, params, save2FileRoute, save2FileArrivalTime, modelArrivalTime)
    params = dict_merge([etpa_result, params])
    save2FileArrivalTime(params)
    return params


