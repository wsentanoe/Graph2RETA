import os
import platform
import sys

if platform.system() == 'Linux':
    file = '/data/MengQingqiang/eta/peta_6_25/'
    sys.path.append(file)
    sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk(file) for name in dirs])

import time, torch, nni, logging
from torch import nn
import torch.nn.functional as F
import numpy as np
from my_utils.utils import ws, mask_loss, MyDataset, get_common_params

from my_utils.transformer_encoder_mask import TransformerEncoder
from my_utils.transformer_encoder_mask import get_transformer_attn_mask

# change 25- > 27
max_len = 27

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class RankpetaDataset(MyDataset):
    def get_input_size(self):
        # change 25- > 27
        # return len(self.global_x[0]) + len(self.unpick_x[0][0]) + 27 + 1

        return len(self.unpick_x[0][0]) + 27 + 1


# mask层定义
def rnn_forwarder(rnn, embedded_inputs, input_lengths, batch_size):
    """
    :param embedded_inputs:
    :param input_lengths:
    :param batch_size:
    :param rnn: RNN instance
    :return: the result of rnn layer,
    """
    """
    packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths.cpu(),
                                               batch_first=rnn.batch_first, enforce_sorted=False)


    _KI_
    # Unpack padding
    # Check Index Naming convention
                                        
    """

    outputs, hidden = rnn(embedded_inputs.to(device))

    # Forward pass through RNN
    try:
        # outputs, hidden = rnn(packed)
        outputs, hidden = rnn(embedded_inputs.to(device))

    except:
        print('lstm encoder:', embedded_inputs.to(device))


    index = input_lengths.to(device)

    # Return output and final hidden state
    if rnn.bidirectional:
        # Optionally, Sum bidirectional RNN outputs
        outputs = outputs[:, :, :rnn.hidden_size] + outputs[:, :, rnn.hidden_size:]

    index2 = index - torch.tensor([1]).to(device)  # 选取操作  Select Operatin
    index1 = torch.tensor(list(range(len(index2)))).to(device)  # 生成第一维度 Generate the First Dimension
    
    index1 = index1.to(torch.int64)
    index2 = index2.to(torch.int64)

    return outputs[index1, index2, :]


# 定义模型 # Define Model
class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.model_name = 'RankPETA'
        self.sort_mode = args["sort_mode"]
        # Features to check
        # ------------------------------------------
        last_x_size = args['last_x_size']
        unpick_up_size = args['unpick_x_size']
        # global_feature_size = args['global_x_size']
        global_feature_size = 0

        # ---------------------------------------
        self.loss_weight = nn.Parameter(torch.tensor([0.5], requires_grad=True))  # 可学习参数 Learnable Parameters
        self.Pos_Table = get_sinusoid_encoding_table(28, args['embedding_dim'])
        self.max_len = 27
        self.embedding_dim = args['embedding_dim']

        # self.pos_beta = 1
        self.number_layer = args['num_block']
        self.hidden_size = args['hidden_size']
        self.n_head = args['num_head']
        self.args = args
        """
        Current State
        """
        # LSTM for latest route
        self.Latest_LSTM = nn.LSTM(input_size=last_x_size, hidden_size=args['hidden_size'],
                                   num_layers=self.number_layer, batch_first=True, bidirectional=True)
        # DNN for Unpick_up Set
        self.unpick_upset_liner1 = nn.Linear(in_features=unpick_up_size, out_features=unpick_up_size // 2)
        self.unpick_upset_liner2 = nn.Linear(in_features=unpick_up_size // 2, out_features=1)

        """
        STattention-based PETA predictor
        """
        self.transformer = TransformerEncoder(n_heads=self.n_head, node_dim=unpick_up_size + self.embedding_dim,
                                              embed_dim=self.hidden_size, n_layers=self.number_layer,
                                              normalization='batch')
        # prediction
        self.liner_ETA = nn.Sequential(
            nn.Linear(in_features=2 * self.hidden_size + self.max_len + global_feature_size + 1 + self.embedding_dim,
                      out_features=(self.hidden_size + self.max_len + self.hidden_size) // 2, bias=False), nn.ELU(),
            nn.Linear(in_features=(self.hidden_size + self.max_len + self.hidden_size) // 2, out_features=32,
                      bias=False), nn.ELU(),
            nn.Linear(in_features=32, out_features=1, bias=False), nn.ReLU())

    def forward(self, last_x, last_len, unpick_x, unpick_len, label_idx, label_order, label_eta, sort_idx, sort_pos):

        """
        #  last_x( bs * 5 * picked_size),
           last_len(bs * 1),
           global_x(bs * feature_size),
           unpick_x( bs * max_len * unpick_x_size)
           unpick_len,
           unpick_geo,
           days_np,  (100,)
           order_np,  (100,25)
           index_np,
           eta_np,
           dic_geo2index

           (bs-> batchsize)

         last_x      : bs * 5 * picked_size
         last_len    : bs * 1

         global_x    : bs * feature_size

         unpick_x    : bs * max_len * unpick_x_size
         unpick_len  : bs * 1

         label_eta   : bs * max_len ( ground truth - Wilson)
         label_idx / label_order: bs * max_len * max_len
         sort_idx  / sort_pos: bs * max_len * max_len

        """
        # loss_sort = 0
        batch_size = unpick_len.shape[0]
        order_info = {}
        # 配置好排序信息 # Configure sorting information ????????
        if self.sort_mode == 'stc' or self.sort_mode == 'none_order':
            order_info = sort_idx
        elif self.sort_mode == 'true_order' or self.sort_mode == 'true_idx':
            order_info = {
                'true_order': label_order,
                'true_idx': label_idx
            }[self.sort_mode]

        else:
            sort_data = self.sort_mode.split('_')[1]
            if sort_data == 'idx':
                order_info = sort_idx
            elif sort_data == 'order' or 'pos':
                order_info = sort_pos

        order_info = self.Pos_Table[order_info.long()].float().to(device)  # .cuda()

        # current state encoder
        latest_route_emb = rnn_forwarder(self.Latest_LSTM, last_x, last_len, batch_size)  # 取最后的输出 Get The Final Output
        unpickup_set_emb = torch.squeeze(
            F.leaky_relu(self.unpick_upset_liner2(F.leaky_relu(self.unpick_upset_liner1(unpick_x.to(device))))))
        current_state = torch.cat(
            [latest_route_emb, unpickup_set_emb, unpick_len.reshape(unpick_len.shape[0], 1).float().to(device)], 1)

        # Prediction Module
        # 先对 global_x 进行扩充为(batch_size*max_len*feature_size)
        # CompactETA pos
        x = torch.cat([unpick_x.to(device), order_info], dim=2)
        attn_mask = get_transformer_attn_mask(max_seq_len=self.max_len, sort_len=unpick_len, batch_size=batch_size)
        attn_mask = attn_mask.to(device)
        transformer_output, _ = self.transformer(x, attn_mask)

        # ETA prediction
        current_state = current_state.unsqueeze(1)
        current_state = current_state.repeat(1, self.max_len, 1)
        # F_input = torch.cat([transformer_output, current_state, global_x, order_info], dim=2)
        F_input = torch.cat([transformer_output, current_state, order_info], dim=2)

        hidden_result = self.liner_ETA(F_input)
        F_output = torch.squeeze(hidden_result)
        pad = nn.ZeroPad2d(padding=(0, self.max_len - F_output.shape[1], 0, 0))
        F_output = pad(F_output)
        mask = label_eta > 0.00001
        F_output *= mask.to(device)
        # 更改了一下损失函数的计算  由于有的不足 25 个padding的 0 所以应该是sum loss除以真实长度
        loss_eta, n = mask_loss(F_output, label_eta)
        return F_output, loss_eta, n

    def model_file_name(self):
        file_name = '+'.join([f'{k}${self.args[k]}' for k in
                              ['datatype', 'hidden_size', 'embedding_dim', 'num_block', 'num_head', 'sort_mode',
                               'train_mode']])
        file_name = f'{file_name}.rankpeta'
        return file_name


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  偶数正弦
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  奇数余弦

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table).to(device)


def save2file(params):
    """
    # Generating CSV with Header and Data #

    """
    import csv
    file_name = ws + '/output/output_RankPETA.csv'
    # 写表头  # Write Header
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        head = ['model', 'datatype', 'sort_mode',
                'rmse', 'mae', 'mape', 'acc@10', 'acc@20', 'acc@30',
                'hidden', 'embedding', 'num_head', 'num_block',
                'stop_epoch', 'train_time', 'test_time',
                'batch', 'lr', 'epoch', 'train_mode',
                'time', 'is_test']
        csv_file.writerow(head)
        f.close()
    # 写数据 # Write data
    with open(file_name, "a", newline='\n') as file:  # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        data = [
            params['modelArrivalTime'], params['datatype'], params['sort_mode'],
            params['rmse'], params['mae'], params['mape'], params['acc@10'], params['acc@20'], params['acc@30'],
            params['hidden_size'], params['embedding_dim'], params['num_head'], params['num_block'],
            params['stop_epoch'], params['train_time'], params['test_time'],
            params['batch_size'], params['lr'], params['num_epoch'], params['train_mode'],
            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), params['is_test']
        ]
        csv_file.writerow(data)


def get_params():
    parser = get_common_params()  # Coming from utils file
    parser.add_argument('--train_ptr', type=int, default=0, metavar='N', help='train pnn or not')
    parser.add_argument("--hidden_size", type=int, default=64, metavar='N',
                        help='hidden layer last_x size (default:64)')
    parser.add_argument("--embedding_dim", type=int, default=64, metavar='N', help='embedding_dim')
    parser.add_argument("--num_block", type=int, default=2, metavar='N', help='num_block')
    parser.add_argument("--num_head", type=int, default=4, metavar='N', help='num_head')
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    logger = logging.getLogger('RankPETA training')
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        params['modelArrivalTime'] = 'RankPETA'
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise