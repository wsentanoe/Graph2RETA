import torch
import torch.nn as nn
import numpy as np
from graph2route.graph2route_pd.encoder import GCNLayer, GATPyG, GATV2
from graph2route.graph2route_pd.decoder import Decoder
from graph2route.graph2route_pd.tft import TFT
import time
import pickle
from utils.utils import preprocess_food


# from graph2route.egat import EGATConv
# import dgl

class Graph2Route(nn.Module):

    def __init__(self, config):
        super(Graph2Route, self).__init__()
        self.config = config
        self.N = config['max_task_num'] + 2  # max nodes
        self.device = config['device']

        # input feature dimension
        self.d_v = config['node_fea_dim']  # dimension of node feature
        self.d_e = config.get('edge_fea_dim', 5)  # dimension of edge feature
        self.d_s = config.get('start_fea_dim', 5)  # dimension of start node feature
        self.d_h = config['hidden_size']  # dimension of hidden size
        self.d_dyn = config.get('dynamic_feature_dim', 2)  # dimension of dynamic feature
        self.d_w = config['worker_emb_dim']  # dimension of worker embedding

        # feature embedding module
        self.worker_emb = nn.Embedding(config['num_worker_pd'], self.d_w)
        self.node_emb = nn.Linear(self.d_v, self.d_h, bias=False)
        self.edge_emb = nn.Linear(self.d_e, self.d_h, bias=False)
        self.start_node_emb = nn.Linear(self.d_s, self.d_h + self.d_v + self.d_dyn)

        # encoding module
        self.gcn_num_layers = config['gcn_num_layers']
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(hidden_dim=self.d_h, aggregation="mean") for _ in range(self.gcn_num_layers)])
        self.graph_gru = nn.GRU(self.N * self.d_h, self.d_h, batch_first=True)
        self.graph_linear = nn.Linear(self.d_h, self.N * self.d_h)
        self.graph_linear_tft = nn.Linear(160, self.N * self.d_h)
        self.graph_lstm = nn.LSTM(self.N * self.d_h, self.d_h, batch_first=True)

        # decoding module
        self.decoder = Decoder(
            self.d_h + self.d_v + self.d_dyn,
            self.d_h + self.d_v + self.d_dyn,
            self.d_w,
            tanh_exploration=10,
            use_tanh=True,
            n_glimpses=1,
            mask_glimpses=True,
            mask_logits=True,
        )

        # GAT
        self.gat_layer_pyg = nn.ModuleList([GATPyG(8, 8, 8) for _ in range(self.gcn_num_layers)])
        self.gat_layer_v2 = nn.ModuleList([
            GATV2(nfeat=8,
                  nhid=8,
                  nclass=8,
                  dropout=0.6,
                  nheads=8,
                  alpha=0.2,
                  device=config['device']) for _ in range(self.gcn_num_layers)])
        """self.egat_layer = EGATConv(
            in_node_feats=self.d_h,
            in_edge_feats=self.d_h,
            out_node_feats=self.d_h,
            out_edge_feats=self.d_h,
            num_heads=3
        )"""

        # TFT
        config_tft = {}
        config_tft['static_variables'] = 1  # Courier ID
        config_tft['time_varying_categoical_variables'] = 1
        config_tft['time_varying_real_variables_encoder'] = 215
        config_tft['time_varying_real_variables_decoder'] = 3
        config_tft['num_masked_series'] = 1
        config_tft['static_embedding_vocab_sizes'] = [1000]
        config_tft['time_varying_embedding_vocab_sizes'] = [1000]
        config_tft['embedding_dim'] = 8
        config_tft['lstm_hidden_dimension'] = 160
        config_tft['lstm_layers'] = 1
        config_tft['dropout'] = 0.05
        config_tft['device'] = self.device
        config_tft['batch_size'] = config['batch_size']
        config_tft['encode_length'] = 8
        config_tft['attn_heads'] = 4
        config_tft['num_quantiles'] = 3
        config_tft['vailid_quantiles'] = [0.1, 0.5, 0.9]
        config_tft['seq_length'] = 12
        self.model_tft = TFT(config_tft)

    def get_init_input(self, t, batch_size, V_val, start_idx_t, V, current_time, V_dt, E_mask_t, V_dispatch_mask, E_ed,
                       E_sd):

        idx_0 = torch.tensor(list(range(batch_size)))
        current_step_x_nodes = V[idx_0, start_idx_t, :]
        current_step_dispatch_time = V_dt.unsqueeze(2)[idx_0, start_idx_t, :]

        start_fea = torch.cat([current_step_x_nodes, current_time, current_step_dispatch_time], dim=1)  # (B, 5)
        decoder_input = self.start_node_emb(start_fea)

        # mask
        V_val_masked = V_val * V_dispatch_mask[:, t, :].unsqueeze(2).expand(V_val.size(0), V_val.size(1), V_val.size(2))
        E_ed_t_masked = E_ed * E_mask_t
        E_sd_t_masked = E_sd * E_mask_t

        return decoder_input, V_val_masked, E_ed_t_masked, E_sd_t_masked

    def adjacency_matrix_to_edge_index(self, adj_matrix):
        num_nodes = adj_matrix.shape[0]
        edge_index = (adj_matrix != 0).nonzero().t()
        filt = torch.ne(edge_index[0], edge_index[1])
        edge_index = edge_index[:, filt]
        return edge_index

    def forward(self, V, V_reach_mask, V_ft, V_pt, V_dt, V_num, V_dispatch_mask, E, E_ed, E_sd, E_mask, start_idx, cou):
        """
        Args:
            V: node features, including coordinates and promised pick-up time (B, N, 3)
            V_ft: latest finish time when new task comes (B, N)
            V_pt: promised pick-up time of nodes (B, N)
            V_dt: dispatch time of nodes (B, N)
            V_num: unfinished task num of each node at each time step (B, T, N)
            V_reach_mask:  mask for reachability of nodes (B, N, N)
            V_dispatch_mask: Input node mask for undispatched nodes (B, T, N)

            E_ed: edge Eulidean distance matrix (B, N, N)
            E_sd: edge geodesic distance matrix (B, N, N)
            E_mask: Input edge mask for undispatched nodes (B, T, N, N)
            E: masked edge features, include edge absolute geodesic distance, edge relative geodesic distance, input spatial-temporal adjacent feature,
             difference of promised pick-up time between nodes, difference of dispatch_time between nodes (B, T, N, N, d_e), here d_e = 5

            start_idx: latest finish node index when new task comes (B, T)
            cou: features of couriers including id, level, velocity and maxload (B, 4)
        :return:
            pointer_log_scores.exp() : rank logits of nodes at each step (B * T, N, N)
            pointer_argmax : decision made at each step (B * T, N)
        """

        B, T, N = V_reach_mask.shape
        d_h, d_v, d_dyn = self.d_h, self.d_v, self.d_dyn

        # batch input
        b_decoder_input = torch.zeros([B, T, d_h + d_v + d_dyn]).to(self.device)
        init_hx = torch.randn(B * T, d_h + d_v + d_dyn).to(self.device)
        init_cx = torch.randn(B * T, d_h + d_v + d_dyn).to(self.device)
        b_V_reach_mask = V_reach_mask.reshape(B * T, N)
        b_V_val = torch.zeros([B, T, N, d_v]).to(self.device)
        b_E_ed_t_masked = torch.zeros([B, T, N, N]).to(self.device)
        b_E_sd_t_masked = torch.zeros([B, T, N, N]).to(self.device)
        b_node_h = torch.zeros([B, T, N, d_h]).to(self.device)
        b_edge_h = torch.zeros([B, T, N, N, d_h]).to(self.device)
        b_V_dy = torch.zeros([B, T, N, d_dyn]).to(self.device)
        b_V_val_unmasked = torch.zeros([B, T, N, d_v]).to(self.device)

        cou = torch.repeat_interleave(cou.unsqueeze(1), repeats=T, dim=1).reshape(B * T, -1)
        embed_cou = torch.cat(
            [self.worker_emb(cou[:, 0].long()), cou[:, 1].unsqueeze(1), cou[:, 2].unsqueeze(1), cou[:, 3].unsqueeze(1)],
            dim=1)
        for t in range(T):
            E_mask_t = torch.FloatTensor(E_mask[:, t, :, :]).to(V.device)

            idx_0, idx_1 = torch.tensor(list(range(B))), start_idx[:, t]
            t_c = V_ft[idx_0, idx_1].unsqueeze(-1)  # (B, 1)
            E_ed_dif = E_ed[idx_0, idx_1]  # (B, N)
            E_sd_dif = E_sd[idx_0, idx_1]  # (B, N)

            # mask edges of undispatched nodes
            E_ed_dif = E_ed_dif * V_dispatch_mask[:, t, :]
            E_sd_dif = E_sd_dif * V_dispatch_mask[:, t, :]

            V_val = torch.cat(
                [V, (V_pt - t_c).unsqueeze(2), (t_c - V_dt).unsqueeze(2), E_ed_dif.unsqueeze(2), E_sd_dif.unsqueeze(2),
                 V_num[:, t, :].unsqueeze(2)], dim=2)  # (B, N, d_v)
            V_dyn = torch.cat([E_ed_dif.unsqueeze(2), E_sd_dif.unsqueeze(2)], dim=2)  # (B, N, d_dyn)

            # mask features of undispatched nodes
            graph_node = V_val * V_dispatch_mask[:, t, :].unsqueeze(2).expand(B, N, d_v)
            graph_node_t = self.node_emb(graph_node)  # (B, N, d_h)
            graph_edge_t = self.edge_emb(E[:, t, :, :])  # (B, N, N, d_h)
            b_node_h[:, t, :, :] = graph_node_t  # b_node_h:(B, T, N, d_h);   graph_node_t: (B, N, d_h)
            b_edge_h[:, t, :, :, :] = graph_edge_t  # e_edge_h: (B, T, N, N, d_h); graph_edge_t: (B, N, N, d_h)

            decoder_input, V_val_masked, E_ed_t_masked, E_sd_t_masked = self.get_init_input(t, B, V_val,
                                                                                            start_idx[:, t], V, t_c,
                                                                                            V_dt, E_mask_t,
                                                                                            V_dispatch_mask, E_ed, E_sd)

            b_decoder_input[:, t, :] = decoder_input  # (B, self.d_h + self.d_v + self.d_dyn )
            b_V_val[:, t, :, :] = V_val_masked
            b_E_ed_t_masked[:, t, :, :] = E_ed_t_masked
            b_E_sd_t_masked[:, t, :, :] = E_sd_t_masked
            b_V_dy[:, t, :, :] = V_dyn
            b_V_val_unmasked[:, t, :, :] = V_val

        # Save b_V_val to file
        """output_dict = {}
        output_dict['b_V_val'] = b_V_val
        with open('node_features.npy', 'wb') as df_file:
            pickle.dump(obj=output_dict, file=df_file)"""

        # encode
        if self.config['spatial_encoder'] == 'gcn':
            for layer in range(self.gcn_num_layers):
                b_node_h, b_edge_h = self.gcn_layers[layer](b_node_h.reshape(B * T, N, d_h),
                                                            b_edge_h.reshape(B * T, N, N, d_h))
        """
         _KI_
       
          GAT START
        """
        if self.config['spatial_encoder'] == 'gatv2':
            batch_node_h_gat = torch.zeros([B * T, N, d_h]).to(self.device)
            E_mask_gat = torch.FloatTensor(E_mask).reshape(B * T, N, N).to(self.device)

            for layer in range(self.gcn_num_layers):
                batch_node_h_gat = self.gat_layer_v2[layer](b_node_h.reshape(B * T, N, d_h), E_mask_gat)

            b_node_h = batch_node_h_gat.reshape(B, T, N, d_h)
        
        """
        _KI_     
        # GAT PyG START
        """
        # 
        if self.config['spatial_encoder'] == 'gat':
            batch_node_h_gat = torch.zeros([B * T, N, d_h]).to(self.device)
            E_mask_gat = torch.FloatTensor(E_mask).reshape(B * T, N, N).to(self.device)

            for layer in range(self.gcn_num_layers):
                for i in range(B * T):
                    edge_index = self.adjacency_matrix_to_edge_index(E_mask_gat[i, :, :])
                    batch_node_h_gat[i, :, :], _ = self.gat_layer_pyg[layer](b_node_h.reshape(B * T, N, d_h)[i, :, :],
                                                                             edge_index,
                                                                             b_edge_h.reshape(B * T, N, N, d_h)[i,
                                                                             edge_index[0], edge_index[1], :])

            b_node_h = batch_node_h_gat.reshape(B, T, N, d_h)

        # EGAT START
        """batch_node_h_egat = torch.zeros([B * T, N, d_h]).to(self.device)
        batch_edge_h_egat = torch.zeros([B * T, N, N, d_h]).to(self.device)
        E_mask_egat = torch.FloatTensor(E_mask).reshape(B * T, N, N).to(self.device)

        for i in range(B * T):
            edge_index = self.adjacency_matrix_to_edge_index(E_mask_egat[i, :, :])
            u, v = edge_index[0], edge_index[1]
            graph = dgl.graph((u, v), num_nodes=N)
            new_node_feats, new_edge_feats = self.egat_layer(graph,
                                                             b_node_h.reshape(B * T, N, d_h)[i, :, :],
                                                             b_edge_h.reshape(B * T, N, N, d_h)[i, u, v, :])
            batch_node_h_egat[i, :, :] = torch.mean(new_node_feats, 1)
            batch_edge_h_egat[i, u, v, :] = torch.mean(new_edge_feats, 1)

        b_node_h = batch_node_h_egat.reshape(B, T, N, d_h)
        b_edge_h = batch_edge_h_egat.reshape(B, T, N, N, d_h)"""
        # EGAT END

        if self.config['temporal_encoder'] == 'gru':
            b_node_h, _ = self.graph_gru(b_node_h.reshape(B, T, -1))
            b_node_h = self.graph_linear(b_node_h)  # （B, T, N * d_h)
        elif self.config['temporal_encoder'] == 'lstm':
            b_node_h, _ = self.graph_lstm(b_node_h.reshape(B, T, -1))
            b_node_h = self.graph_linear(b_node_h)  # （B, T, N * d_h)
        elif self.config['temporal_encoder'] == 'tft':
            b_node_h = self.model_tft(cou[:, 0].reshape(B, T, 1), b_node_h.reshape(B, T, -1))
            b_node_h = self.graph_linear_tft(b_node_h)  # （B, T, N * d_h)

        inputs = torch.cat(
            [b_node_h.reshape(B * T, N, d_h), b_V_val.reshape(B * T, N, d_v), b_V_dy.reshape(B * T, N, d_dyn)],
            dim=2).permute(1, 0, 2).contiguous().clone()  #
        enc_h = torch.cat(
            [b_node_h.reshape(B * T, N, d_h), b_V_val.reshape(B * T, N, d_v), b_V_dy.reshape(B * T, N, d_dyn)],
            dim=2).permute(1, 0, 2).contiguous().clone()

        #  decode
        (pointer_log_scores, pointer_argmax, final_step_mask) = \
            self.decoder(
                b_decoder_input.reshape(B * T, d_h + d_v + d_dyn),
                inputs.reshape(N, T * B, d_h + d_v + d_dyn),
                (init_hx, init_cx),
                enc_h.reshape(N, T * B, d_h + d_v + d_dyn),
                b_V_reach_mask, b_node_h.reshape(B * T, N, d_h),
                b_V_val.reshape(B * T, N, d_v), b_E_ed_t_masked.reshape(B * T, N, N),
                b_E_sd_t_masked.reshape(B * T, N, N), embed_cou, start_idx.reshape(B * T), self.config)

        return pointer_log_scores.exp(), pointer_argmax, b_V_val_unmasked

    def model_file_name(self):
        t = time.time()
        file_name = '+'.join([f'{k}-{self.config[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}.gcnru-pd_{t}'
        return file_name


from torch.utils.data import Dataset


#  Dataset Function Changes

class Graph2RouteDataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict,  # parameters dict
    ) -> None:
        super().__init__()
        if mode not in ["train", "val", "test", ]:  # "validate"
            raise ValueError
        path_key = {'train': 'train_path', 'val': 'val_path', 'test': 'test_path', }[mode]

        # For node fearures access
        node_feature_path_key = \
        {'train': 'node_features_train', 'val': 'node_features_val', 'test': 'node_features_test', }[mode]
        node_features_path = params[node_feature_path_key]

        path = params[path_key]
        self.device = params['device']
        self.data = np.load(path, allow_pickle=True).item()
        #  Add path of node_feature_file
        self.node_features = np.load(node_features_path, allow_pickle=True)
        V = self.data['V']
        V_ft = self.data['V_ft']
        V_dt = self.data['V_dt']
        V_dispatch_mask = self.data['V_dispatch_mask']
        cou = self.data['cou']
        V_val = self.node_features['V_val']

        # Hyperparameter
        pad_value = 26
        self.unpick_x, self.unpick_len, self.last_x, self.last_len, self.eta_np, self.order_np, self.index_np = preprocess_food(
            V, V_ft, V_dt, V_dispatch_mask, cou, V_val, pad_value)

        self.size_dict = {
            'last_x_size': 13,#len(self.last_x[0][0]),
            'unpick_x_size': 12#len(self.unpick_x[0][0])
        }

    def get_size(self):
        input_size = len(self.unpick_x[0][0]) + 25 + 1
        sort_x_size = len(self.unpick_x[0][0])
        return input_size, sort_x_size

    # from abc import abstractmethod
    # @abstractmethod
    def get_input_size(self):  # redefine this function for different method
        return len(self.unpick_x[0][0]) + 25 + 1

    def size(self):
        self.size_dict['input_size'] = self.get_input_size()
        return self.size_dict

    def __len__(self):
        return len(self.data['nodes_num'])

    def __getitem__(self, index):
        E_ed = self.data['E_ed'][index]
        E_sd = self.data['E_sd'][index]
        E_mask = self.data['E_mask'][index]

        V = self.data['V'][index]  # nodes features
        V_reach_mask = self.data['V_reach_mask'][index]
        V_pt = self.data['V_pt'][index]
        V_ft = self.data['V_ft'][index]
        V_num = self.data['V_num'][index]  # num of order at each loc at each step
        V_dispatch_mask = self.data['V_dispatch_mask'][index]
        V_dt = self.data['V_dt'][index]

        label_len = self.data['label_len'][index]
        label = self.data['label'][index]

        start_idx = self.data['start_idx'][index]

        pt_dif = self.data['pt_dif'][index]
        dt_dif = self.data['dt_dif'][index]
        cou = self.data['cou'][index]
        A = self.data['A'][index]

        #  These variables are for RankETPA Model
        unpick_x = self.unpick_x[index]
        unpick_len = self.unpick_len[index]
        last_x = self.last_x[index]
        last_len = self.last_len[index]
        eta_np = self.eta_np[index]
        order_np = self.order_np[index]
        index_np = self.index_np[index]

        return E_ed, V, V_reach_mask, label_len, label, V_pt, V_ft, start_idx, \
            E_sd, V_dt, V_num, E_mask, V_dispatch_mask, pt_dif, dt_dif, cou, A, last_x, last_len, unpick_x, unpick_len, \
            eta_np, order_np, index_np, index


# ---Log--
from utils.utils import save2file_meta


def save2file(params):
    from utils.utils import ws
    file_name = ws + f'/output/food_pd/{params["modelRoute"]}.csv'
    # 写表头
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max',
        # mdoel parameters
        'modelRoute', 'hidden_size',
        # training set
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time',
        # metric result
        'lsd', 'lmd', 'krc', 'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',
    ]
    save2file_meta(params, file_name, head)