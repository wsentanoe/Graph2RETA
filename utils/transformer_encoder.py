import torch
import torch.nn as nn
import math
MAX_ORDER_NUM = 25


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, inputs):
        outputs = self.module(inputs)
        outputs += inputs
        return outputs


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 n_heads,
                 input_dim,
                 embed_dim=None,
                 val_dim=None,
                 key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            std = 1. / math.sqrt(param.size(-1))
            nn.init.uniform_(param, -std, std)

    def forward(self, inputs):
        q = inputs
        h = inputs

        batch_size, seq_len, input_dim = h.size()
        n_query = q.size(1)

        h_flat = h.contiguous().view(-1, input_dim)
        q_flat = q.contiguous().view(-1, input_dim)

        k_shape = (self.n_heads, batch_size, seq_len, -1)
        q_shape = (self.n_heads, batch_size, n_query, -1)
        Q = torch.matmul(q_flat, self.W_query).view(q_shape)
        K = torch.matmul(h_flat, self.W_key).view(k_shape)
        V = torch.matmul(h_flat, self.W_val).view(k_shape)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        attn = torch.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)
        outputs = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return outputs


class Normalization(nn.Module):

    def __init__(self, seq_len):
        super(Normalization, self).__init__()
        self.normalizer = nn.BatchNorm1d(seq_len)
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            std = 1. / math.sqrt(param.size(-1))
            nn.init.uniform_(param, -std, std)

    def forward(self, inputs: torch.Tensor):
        return self.normalizer(inputs)


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(MAX_ORDER_NUM),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim),
            ),
            Normalization(MAX_ORDER_NUM)
        )


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            n_heads=8,
            embed_dim=64,
            n_layers=6,
            input_dim=None,
            feed_forward_hidden=256
    ):
        super(TransformerEncoder, self).__init__()

        self.transformer = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        ))

    def forward(self, x):
        x = self.transformer(x)
        return x, (x.mean(1), x.mean(1))
