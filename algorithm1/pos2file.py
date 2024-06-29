# -*- coding: utf-8 -*-
import numpy as np
import time,torch, nni, argparse, os, json, logging, platform, sys
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from my_utils1.utils import ws, get_dataset_path, batch_file_name
from algorithm1.pointer_net import PointNet
import warnings
warnings.filterwarnings("ignore")

max_len = 25
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, last_x, last_len, global_x, unpick_x, unpick_len, label_idx, label_order, label_eta, sample_num):
        super(MyDataset, self).__init__()
        self.len = len(label_eta) if sample_num == -1 else sample_num
        self.last_x = last_x[:self.len]
        self.last_len = last_len[:self.len]
        self.global_x = global_x[:self.len]
        self.unpick_x = unpick_x[:self.len]
        self.unpick_len = unpick_len[:self.len]
        self.label_idx = label_idx[:self.len]
        self.label_eta = label_eta[:self.len]
        self.label_order = label_order[:self.len]

    def __len__(self):
        # return len(self.label_eta)
        return self.len

    def __getitem__(self, index):
        return torch.from_numpy(self.last_x[index]).float(), \
               self.last_len[index], \
               torch.from_numpy(self.global_x[index]).float(), \
               torch.from_numpy(self.unpick_x[index]).float(), \
               self.unpick_len[index], \
               torch.from_numpy(self.label_idx[index]).int(), \
               torch.from_numpy(self.label_order[index]).int(),\
               torch.from_numpy(self.label_eta[index]).float()

    def get_size(self):
        input_size = len(self.global_x[0]) + len(self.unpick_x[0][0]) + 25
        sort_x_size = len(self.unpick_x[0][0])
        return input_size, sort_x_size


def get_model_para(path):
    # get the parameter and performance dict from the file path
    # order_Ken0.391_HR0.808_LSD3.677_hidden16_t13.46.pnn_model
    file_name = ''
    if os.path.isfile(path):
        file_name = os.path.split(path)[1]

    file_name = file_name.split('_')
    para = {}
    for x in file_name:
        if 'hidden' in x:
            hidden = int(x.replace('hidden', ''))
            para['hidden_size'] = hidden
        if 'Ken' in x:
            para['krc'] = float(x.lstrip('Ken'))
    return para


def get_best_model_path(data_file):
    pnn_models = batch_file_name(data_file, '.pnn_model')
    assert len(pnn_models) != 0, "route predictor does not exist, please train it first!"
    krc_dict = {path: get_model_para(path)['krc'] for path in pnn_models}
    best = max(krc_dict, key=krc_dict.get)
    return best


def train_test(train_loader, test_loader, model, is_test):
    train_idx = []
    train_pos = []
    test_idx = []
    test_pos = []
    pbar = tqdm(total=len(train_loader))
    for batch_idx, data in enumerate(train_loader):
        if is_test and batch_idx > 5: break
        pbar.update(1)
        last_x, last_len, global_x, unpick_x, \
            unpick_len, label_idx, label_order, label_eta = [d.to(device) for d in data]
        log_pointer_score, argmax_pointer, pos, mask = model(unpick_len, unpick_x, get_pos=True)

        train_idx += argmax_pointer.cpu().numpy().tolist()
        train_pos += pos.cpu().numpy().tolist()
    pbar.close()

    pbar = tqdm(total=len(test_loader))
    for batch_idx, data in enumerate(test_loader):
        if is_test and batch_idx > 5: break
        pbar.update(1)
        last_x, last_len, global_x, unpick_x, \
            unpick_len, label_idx, label_order, label_eta = [d.to(device) for d in data]
        log_pointer_score, argmax_pointer, pos, mask = model(unpick_len, unpick_x, get_pos=True)

        test_idx += argmax_pointer.cpu().numpy().tolist()
        test_pos += pos.cpu().numpy().tolist()
    pbar.close()

    train_idx = np.array(train_idx)
    train_pos = np.array(train_pos)
    test_idx = np.array(test_idx)
    test_pos = np.array(test_pos)
    return train_idx, train_pos, test_idx, test_pos


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(param):
    torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    print('device:' + str(device))
    if str(device) == 'cuda':
        print('current cuda id:',str(torch.cuda.current_device()))

    train_path, test_path = get_dataset_path(param)
    data_file = os.path.dirname(train_path)
    fout = data_file + f'/PnnOutput_{param["datatype"]}.pkl'
    print('data: ' + train_path)

    last_x, last_len, global_x, unpick_x, unpick_len, unpick_geo, days_np, order_np, index_np, eta_np, dic_geo2index = np.load(train_path, allow_pickle=True)
    train_set = MyDataset(last_x, last_len, global_x, unpick_x, unpick_len, index_np, order_np, eta_np, sample_num=-1)
    train_loader = DataLoader(dataset=train_set, batch_size=512, shuffle=False)

    last_x, last_len, global_x, unpick_x, unpick_len, unpick_geo, days_np, order_np, index_np, eta_np, dic_geo2index = np.load(test_path, allow_pickle=True)
    test_set = MyDataset(last_x, last_len, global_x, unpick_x, unpick_len, index_np, order_np, eta_np, sample_num=-1)
    test_loader = DataLoader(dataset=test_set, batch_size=512, shuffle=False)

    print(f"#train:{len(train_set)} | #test:{len(test_set)}")
    input_size, sort_x_size = train_set.get_size()
    print(f"input_size:{input_size}, sort_x_size:{sort_x_size}")

    ptr_model_path = get_best_model_path(data_file)

    print('model path: ' + ptr_model_path)
    ptr_para = get_model_para(ptr_model_path)
    model = PointNet({'hidden_size': ptr_para['hidden_size'], 'sort_x_size': sort_x_size})
    model.load_state_dict(torch.load(ptr_model_path, map_location='cpu'))
    model.eval()
    model.to(device)

    is_test = param.get('is_test', False)
    pickle.dump(train_test(train_loader, test_loader, model, is_test), open(fout, 'wb'))


if __name__ == '__main__':
    params = {
        'datatype': 'order',
        'dataset': 'shanghai-b10_0-120_3-25',
        'is_test': False,
    }
    main(params)


