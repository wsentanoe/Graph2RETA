import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

import numpy as np

class GATPyG(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads, edge_dim=dim_in)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1, edge_dim=dim_in)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index, edge_feats):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index, edge_feats)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index, edge_feats)
        return h, F.log_softmax(h, dim=1)

# GATv2 - Manual Implementation
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, device=None):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W1 = nn.Linear(in_features, out_features, bias=False)
        self.W2 = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(out_features, 1, bias=False)

        """self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)"""

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.device = device

    def forward(self, h, adj):
        # Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh1 = self.W1(h)
        Wh2 = self.W2(h)
        e = self._prepare_attentional_mechanism_input(Wh1, Wh2)

        zero_vec = -9e15 * torch.ones_like(e).to(self.device)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh1)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh1, Wh2):
        B, N, H = Wh1.shape

        # Aggregate the Whs
        Wh1 = Wh1.repeat(1, N, 1)
        Wh2 = Wh2.repeat_interleave(N, dim=1)
        Wh_sum = Wh1 + Wh2
        Wh_sum = Wh_sum.view(B, N, N, H)

        # Apply the leakyrelu, and then multiply it with a
        e = self.a(self.leakyrelu(Wh_sum))

        return e.squeeze(-1)

        """# Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        print("Shape Wh1: ", Wh1.shape)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        print("Shape Wh2: ", Wh2.shape)
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)"""

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class GATV2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, device):
        """Dense version of GAT."""
        super(GATV2, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, device=device) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=2)


class BatchNormNode(nn.Module):

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):

        x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
        return x_bn


class BatchNormEdge(nn.Module):

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        e_trans = e.transpose(1, 3).contiguous()  # Reshape input: (batch_size, num_nodes, num_nodes, hidden_dim)
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()  # Reshape to original
        return e_bn


class NodeFeatures(nn.Module):

    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, edge_gate):

        Ux = self.U(x)  # B x V x H
        Vx = self.V(x)  # B x V x H
        Vx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        gateVx = edge_gate * Vx  # B x V x V x H
        if self.aggregation == "mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation == "sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        return x_new


class EdgeFeatures(nn.Module):

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, e):

        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)  # Extend Vx from "B x V x H" to "B x V x 1 x H"
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        e_new = Ue + Vx + Wx
        return e_new


class GCNLayer(nn.Module):

    def __init__(self, hidden_dim, aggregation="sum"):
        super(GCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):

        e_in = e
        x_in = x
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x V x V x H
        # Compute edge gates
        # edge_gate = F.sigmoid(e_tmp)
        edge_gate = torch.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate)
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new