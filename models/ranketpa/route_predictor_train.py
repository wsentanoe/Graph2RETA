# -*- coding: utf-8 -*-
import os
from pprint import pprint
import argparse
import logging
import nni
import time

import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import warnings

warnings.filterwarnings("ignore")

from models.ranketpa.pointer_net import PointNet
from utils import ws, EarlyStop, get_dataset_path, AverageMeter, dir_check
from evaluation.eval import sort_eval

"""
Original route predictor of the ranketpa model (not used)
"""


class my_Dataset(Dataset):
    def __init__(self, unpick, lengths, index):
        self.seqs = unpick
        self.labels = index
        self.lengths = lengths
        self.input_dim = unpick.shape[2]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        label = torch.LongTensor(self.labels[index])
        len_order = self.lengths[index]

        seq = torch.FloatTensor(seq)
        return seq, len_order, label


def get_sample(unpick_x, unpick_len, global_x, start, end):
    global_x = np.expand_dims(global_x, axis=1).repeat(25, axis=1)[start:end]
    unpick_len = unpick_len.reshape(unpick_len.shape[0], 1, 1).repeat(25, axis=1)[start:end]
    X = np.concatenate([global_x, unpick_x[start:end], unpick_len], axis=2)
    return X


def get_dataset(params, only_x=False):

    train_path, test_path = get_dataset_path(params)
    print('pnn using data: ' + train_path)

    # train, valid
    last_x, last_len, global_x, unpick_x, unpick_len, unpick_geo, \
        days_np, order_np, index_np, eta_np, dic_geo2index = np.load(train_path, allow_pickle=True)

    flag = int(len(unpick_len) * 0.8)
    if only_x:
        train_set = my_Dataset(unpick_x[0:flag], unpick_len[0:flag], index_np[0:flag])
        valid_set = my_Dataset(unpick_x[flag:len(unpick_len)], unpick_len[flag:len(unpick_len)], index_np[flag:len(unpick_len)])
    else:
        x_train = get_sample(unpick_x, unpick_len, global_x, 0, flag)
        x_valid = get_sample(unpick_x, unpick_len, global_x, flag, len(unpick_len))
        train_set = my_Dataset(x_train, unpick_len[0:flag], index_np[0:flag])
        valid_set = my_Dataset(x_valid, unpick_len[flag:len(unpick_len)], index_np[flag:len(unpick_len)])

    # test
    last_x, last_len, global_x, unpick_x, unpick_len, unpick_geo, \
        days_np, order_np, index_np, eta_np, dic_geo2index = np.load(test_path, allow_pickle=True)

    if only_x:
        test_set = my_Dataset(unpick_x, unpick_len, index_np)
    else:
        x_test = get_sample(unpick_x, unpick_len, global_x, 0, len(unpick_len))
        test_set = my_Dataset(x_test, unpick_len, index_np)

    print(f"#train:{len(train_set)} | #test:{len(test_set)}")
    input_len, input_size = train_set.__len__(), train_set.input_dim
    print(f"input_size:{input_len}, sort_x_size:{input_size}")

    return train_set, valid_set, test_set, input_size


def main(params):
    cuda_id = params.get('cuda_id', None)
    if cuda_id != None:
        torch.cuda.set_device(cuda_id)
    use_cuda = not params['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    cudnn.benchmark = True if use_cuda else False

    pprint(params)

    train_set, valid_set, test_set, input_size = get_dataset(params, only_x=True)
    train_loader = DataLoader(dataset=train_set, batch_size=params['batch_size'], shuffle=True, num_workers=params['workers'])
    valid_loader = DataLoader(dataset=valid_set, batch_size=512, shuffle=True, num_workers=params['workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=512, shuffle=False, num_workers=params['workers'])

    args_model = {
        'hidden_size': params['emb_dim'],
        'sort_x_size': input_size
    }
    model = PointNet(args_model).to(device)

    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    train_loss, valid_loss, test_loss = AverageMeter(), AverageMeter(), AverageMeter()
    train_krc, valid_krc, test_krc = AverageMeter(), AverageMeter(), AverageMeter()
    train_lsd, valid_lsd, test_lsd = AverageMeter(), AverageMeter(), AverageMeter()
    train_hr3, valid_hr3, test_hr3 = AverageMeter(), AverageMeter(), AverageMeter()

    max_valid_krc = 0
    best_epoch = 0
    early_stop = EarlyStop(mode='maximize', patience=params['early_stop'])

    eval_batch = len(train_loader) // 10    # evaluate the model  about 10 times for each epoch
    # eval_batch = 10
    if eval_batch == 0: eval_batch = len(train_loader)
    print('Evaluate the model for each %s batch' % eval_batch)
    model_name = '+'.join([f'{k}${params[k]}' for k in ['datatype', 'emb_dim']])    # include paramters name
    model_path = ws + f'/data/rank_model/{model_name}.pnn'
    dir_check(model_path)
    train_time = time.time()
    for epoch in range(params['epochs']):
        if early_stop.stop_flag:
            break
        # Train
        model.train()
        for batch_idx, (seq, length, target) in enumerate(train_loader):
            seq, length, target = seq.to(device), length.to(device), target.to(device)
            log_pointer_score, argmax_pointer, pos, mask = model(length, seq, get_pos=False)

            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.cross_entropy(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), seq.size(0))

            metrics = sort_eval(argmax_pointer, target, length, length)
            train_krc.update(metrics['krc'], metrics['batch_size'])
            train_lsd.update(metrics['lsd'], metrics['batch_size'])
            train_hr3.update(metrics['hr'], metrics['batch_size'])

            if (batch_idx+1) % eval_batch == 0:
                print('Epoch {}: Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKendall: {:.4f}%'
                      .format(epoch,
                              batch_idx * len(seq),
                              len(train_loader.dataset),
                              100. * batch_idx / len(train_loader),
                              train_loss.avg, 100. * train_krc.avg))

        # Validation
        model.eval()
        for seq, length, target in valid_loader:
            seq, length, target = seq.to(device), length.to(device), target.to(device)

            log_pointer_score, argmax_pointer, pos, mask = model(length, seq, get_pos=False)
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.cross_entropy(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
            valid_loss.update(loss.item(), seq.size(0))
            metrics = sort_eval(argmax_pointer, target, length, length)

            valid_hr3.update(metrics['hr'], metrics['batch_size'])
            valid_krc.update(metrics['krc'], metrics['batch_size'])
            valid_lsd.update(metrics['lsd'], metrics['batch_size'])

        print('Epoch {}: [Valid]  Loss: {:.6f}\tKendall: {:.4f}%\tLSD: {:.6f}\tHR@3: {:.4f}%'
              .format(epoch,
                      valid_loss.avg,
                      100. * valid_krc.avg,
                      valid_lsd.avg,
                      100. * valid_hr3.avg))
        nni.report_intermediate_result(valid_krc.avg)
        early_stop.append(valid_krc.avg)
        if valid_krc.avg > max_valid_krc:
            max_valid_krc = valid_krc.avg
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
        print('Now best epoch: {}\tbest_Kendall: {:.4f}%'.format(best_epoch, 100. * max_valid_krc))
        if params['is_test']:
            early_stop.stop_flag = True
            break

    train_time = time.time() - train_time

    # Test
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for seq, length, target in test_loader:
        seq, length, target = seq.to(device), length.to(device), target.to(device)

        log_pointer_score, argmax_pointer, pos, mask = model(length, seq, get_pos=False)
        unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
        loss = F.cross_entropy(unrolled, target.view(-1), ignore_index=-1)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        test_loss.update(loss.item(), seq.size(0))

        metrics = sort_eval(argmax_pointer, target, length, length)
        test_hr3.update(metrics['hr'], metrics['batch_size'])
        test_krc.update(metrics['krc'], metrics['batch_size'])
        test_lsd.update(metrics['lsd'], metrics['batch_size'])

    print('\nTest result:\nKendall: {:.4f}%\tHR@3: {:.1f}\tLSD: {:.4f}'
          .format(100. * test_krc.avg, 100. * test_hr3.avg, test_lsd.avg))

    # save the model
    train_path, test_path = get_dataset_path(params)
    params['PNN_model_dir'] = os.path.dirname(train_path)
    if not os.path.isdir(params['PNN_model_dir']):
        os.mkdir(params['PNN_model_dir'])

    # pnn_model
    model_name = '%s_Ken%.3f_HR%.3f_LSD%.3f_hidden%d_t%s.pnn_model' % \
                 (params['datatype'],
                  test_krc.avg,
                  test_hr3.avg,
                  test_lsd.avg,
                  params['emb_dim'],
                  str(time.strftime("%H.%M", time.localtime()))
                  )

    torch.save(model.state_dict(), params['PNN_model_dir'] + '/rank_model/' + model_name)
    print(params['PNN_model_dir'] + '/' + model_name)
    print('**********model saved**********')
    print('training time: ' + str(train_time))
    nni.report_final_result(test_krc.avg)
    return 0


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Route Predictor Sorting Unpicked Packages')
    parser.add_argument('--is_test', type=bool, default=False, help='test the code')
    parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension (default: 8)')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop at')

    parser.add_argument('--datatype', default='order', type=str, help='datatype')
    parser.add_argument('--dataset', default='shanghai-b1_0-120_1-25', type=str, help='dataset')#artificial-4method-1111_0-120, 22146_b1_0-120, shanghai-1_b1_5-120

    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args, _ = parser.parse_known_args()

    return args


if __name__ == '__main__':
    logger = logging.getLogger('collect training')
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
