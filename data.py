import pickle
import numpy as np
from my_utils.utils import ws
train_path = ws + f'/data/train_order.npy'
test_path = ws + f'/data/test_order.npy'


def cut(path, num_samples):
    datas = np.load(path, allow_pickle=True)
    datas = list(datas)
    for i in range(len(datas)):
        if type(datas[i]) == type({}):
            datas[i] = np.array([])
            continue
        datas[i] = datas[i][:num_samples]
    datas = tuple(datas)
    with open(path, 'wb') as f:
        pickle.dump(datas, f)

cut(train_path, 100)
cut(test_path, 50)
with open(train_path, 'rb') as f:
    dd = pickle.load(f)
pass
