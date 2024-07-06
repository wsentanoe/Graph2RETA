# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import math

"""
    Evaluation Functions used in route_predictor_train.py

"""
def eta_eval(pred, label, metric='mae'):
    mask = label > 0
    label = label.masked_select(mask)
    pred = pred.masked_select(mask)
    if metric == 'mae': result = nn.L1Loss()(pred, label).item()
    if metric == 'mse': result = nn.MSELoss()(pred, label).item()
    if metric == 'rmse':
        mse = nn.MSELoss()(pred, label).item()
        result = np.sqrt(mse)
    if metric == 'mape': result = torch.abs((pred - label) / label).mean().item()
    if 'acc' in metric:  # calculate the Hit@10min, Hit@20min
        k = int(metric.split('@')[1])
        tmp = torch.abs(pred - label) < k
        result = torch.sum(tmp).item() / tmp.shape[0]
        #result = result * 100
    n = mask.sum().item()
    return result, n

def hit_rate(pred, label, lab_len, top_n=5):
    """
    Get the top-n hit rate of the prediction
    :param lab_len:
    :param pred:
    :param label:
    :param top_n:
    :return:
    """
    label_len = lab_len
    eval_num = min(top_n, label_len)
    hit_num = len(set(pred[:eval_num]) & set(label[:eval_num]))
    hit_rate = hit_num / eval_num
    return hit_rate


def kendall_rank_correlation(pred, label, label_len):
    """
    caculate the kendall rank correlation between pred and label, note that label set is contained in the pred set
    :param label_len:
    :param pred:
    :param label:
    :return:
    """

    def is_concordant(i, j):
        return 1 if (label_order[i] < label_order[j] and pred_order[i] < pred_order[j]) or (
                label_order[i] > label_order[j] and pred_order[i] > pred_order[j]) else 0

    if label_len == 1: return 1

    label = label[:label_len]
    not_in_label = set(pred) - set(label)
    # get order dict
    pred_order = {d: idx for idx, d in enumerate(pred)}
    label_order = {d: idx for idx, d in enumerate(label)}
    for o in not_in_label:
        label_order[o] = len(label)

    n = len(label)
    # compare list 1: compare items between labels
    lst1 = [(label[i], label[j]) for i in range(n) for j in range(i + 1, n)]
    # compare list 2: compare items between label and pred
    lst2 = [(i, j) for i in label for j in not_in_label]

    hit_lst = [is_concordant(i, j) for i, j in (lst1 + lst2)]
    # todo_: add the weight here
    hit = sum(hit_lst)
    not_hit = len(hit_lst) - hit
    result = (hit - not_hit) / (len(lst1) + len(lst2))
    return result


def _sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def idx_weight(i, mode='linear'):
    if mode == 'linear': return 1 / (i + 1)
    if mode == 'exp': return math.exp(-i)
    if mode == 'sigmoid': return _sigmoid(5 - i)  # 5 means we focuse on the top 5
    if mode == 'no_weight': return 1
    if mode == 'log': return 1 / math.log(2 + i)  # i is start from 0


def location_deviation(pred, label, label_len, mode='square'):
    label = label[:label_len]

    n = len(label)
    # get the location in list 1
    idx_1 = [idx for idx, x in enumerate(label)]
    # get the location in list 2
    idx_2 = [pred.index(x) for x in label]

    # caculate the distance
    idx_diff = [math.fabs(i - j) for i, j in zip(idx_1, idx_2)]
    weights = [idx_weight(idx, 'no_weight') for idx in idx_1]

    result = list(map(lambda x: x ** 2, idx_diff)) if mode == 'square' else idx_diff
    return sum([diff * w for diff, w in zip(result, weights)]) / n


def edit_distance(pred, label, label_len):
    import edit_distance
    label = label[:label_len]
    pred = pred[:label_len]
    return edit_distance.SequenceMatcher(pred, label).distance()


def sort_eval(prediction, label, label_len, input_len, args={}):
    """
    evaluate the prediction result
    :param args:
    :param input_len:
    :param label_len:
    :param prediction:
    :param label:
    :return:
    """
    def tensor2lst(x):
        try:
            return x.cpu().numpy().tolist()
        except:
            return x

    prediction, label, label_len, input_len = [tensor2lst(x) for x in [prediction, label, label_len, input_len]]

    # process the prediction
    pred = []
    for p, inp_len in zip(prediction, input_len):
        input = set(range(inp_len))
        tmp = list(filter(lambda pi: pi in input, p))
        pred.append(tmp)

    result = {}
    result.update(args)
    result['hr'] = np.array(
        [hit_rate(pre, lab, lab_len, args.get('top_n', 3)) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
    result['hr5'] = np.array(
        [hit_rate(pre, lab, lab_len, 5) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
    result['krc'] = 0
    result['lsd'] = 0
    result['lmd'] = 0

    result['krc'] = np.array(
        [kendall_rank_correlation(pre, lab, lab_len) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
    result['lsd'] = np.array(
        [location_deviation(pre, lab, lab_len, 'square') for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
    result['lmd'] = np.array(
        [location_deviation(pre, lab, lab_len, 'mean') for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
    result['batch_size'] = len(pred)
    return result


def metrics_to_str(metrics={}):
    # str_ = f'hr@3:{metrics["hr"]:0.4f}\t kendall rank correlation:{metrics["krc"]:0.4f}\t  location square deviation:{metrics["lsd"]:0.4f}\t location mean deviation:{metrics["lmd"]:0.4f}'
    str_ = f'hr@3:{metrics["hr"]:0.4f}\t krc:{metrics["krc"]:0.4f}\t  lsd:{metrics["lsd"]:0.4f}\t'

    return str_


if __name__ == '__main__':
    for i in range(10):
        pred = [i for i in range(10)]
        np.random.shuffle(pred)
        label = [i for i in range(10)]
        np.random.shuffle(label)
        distance = edit_distance(pred, label, 2)
        print('pred:', pred)
        print('label:', label)
        print('distance:', distance)
        print('-' * 50)

    pass
