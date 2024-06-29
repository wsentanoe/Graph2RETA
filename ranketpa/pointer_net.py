# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import sys
import os
import platform
file = 'C:/Study/Research/graph2route/pickup_route_prediction/' if platform.system() == 'Windows' else '/data/MengQingqiang/eta/peta_6_25/'
sys.path.append(file)
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk(file) for name in dirs])
#-------------------------------------------------------------------------------------------------------------------------#


import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor


import warnings
warnings.filterwarnings("ignore")

from torch.multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from typing import Union

def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        # mask = mask.float()
        mask = mask.bool()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_max(vector: torch.Tensor,mask: torch.Tensor,dim: int,keepdim: bool = False,min_val: float = -1e7) -> (torch.Tensor, torch.Tensor):
    """
    To calculate max along certain dimensions on masked values
    Parameters
    ----------
    vector : ``torch.Tensor``
        The vector to calculate max, assume unmasked parts are already zeros
    mask : ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    dim : ``int``
        The dimension to calculate max
    keepdim : ``bool``
        Whether to keep dimension
    min_val : ``float``
        The minimal value for paddings
    Returns
    -------
    A ``torch.Tensor`` of including the maximum values.
    """
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super(LSTMEncoder, self).__init__()

        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_layers =  num_layers
        self.embedding_dim =  embedding_dim
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)


        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)


    def forward(self, embedded_inputs, input_lengths, max_len):
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths.cpu(), batch_first=self.batch_first,enforce_sorted=False)
        # Forward pass through RNN
        try:
            outputs, hidden = self.rnn(packed)
        except:
            print('lstm encoder:', embedded_inputs)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
        # Unpack函数只能padding至当前batch最大长度，需继续pad至全局最大长度
        extra_padding_size = max_len - outputs.shape[1]
        outputs = nn.functional.pad(outputs, [0,0,0,extra_padding_size,0,0], mode="constant", value=0)

        # Return output and final hidden state
        if self.bidirectional:
            # Optionally, Sum bidirectional RNN outputs
            outputs = torch.cat((outputs[:, :, :self.hidden_size], outputs[:, :, self.hidden_size:]), dim=2)
        batch_size = embedded_inputs.size(0)
        h_n, c_n = hidden
        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        c_n = c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        if self.bidirectional:
            f = (h_n[-1, 0, :, :].squeeze(), c_n[-1, 0, :, :].squeeze())
            b =  (h_n[-1, 1, :, :].squeeze(), c_n[-1, 1, :, :].squeeze())
            hidden = (torch.cat((f[0], b[0]), dim=1), torch.cat((f[1], b[1]), dim=1))
        else:
            hidden = (h_n[-1, 0, :, :].squeeze(), c_n[-1, 0, :, :].squeeze())


        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 tanh_exploration,
                 use_tanh,
                 n_glimpses=1,
                 mask_glimpses=True,
                 mask_logits=True,
                 geo_vocab_size = 10):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = 'greedy'  # Needs to be set explicitly before use


        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = Attention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(dim=1)



    def update_mask(self, mask, selected):
        def mask_modify(mask):
            all_true = mask.all(1)
            mask_mask = torch.zeros_like(mask)
            mask_mask[:, -1] = all_true
            return mask.masked_fill(mask_mask, False)

        result_mask = mask.clone().scatter_(1, selected.unsqueeze(-1), True)
        return mask_modify(result_mask)

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):

        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask

        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)

        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()

        if not self.mask_logits:
            # If self.mask_logits, this would be redundant, otherwise we must mask to make sure we don't resample
            # Note that as a result the vector of probs may not sum to one (this is OK for .multinomial sampling)
            # But practically by not masking the logits, a model is learned over all sequences (also infeasible)
            # while only during sampling feasibility is enforced (a.k.a. by setting to 0. here)
            probs[logit_mask] = 0.
            # For consistency we should also mask out in log_p, but the values set to 0 will not be sampled and
            # Therefore not be used by the reinforce estimator

        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None):

        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits
        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)

        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)
            # For the glimpses, only mask before softmax so we have always an L1 norm 1 readout vector
            if mask_glimpses:
                logits[logit_mask] = -np.inf
            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[logit_mask] = -np.inf

        return logits, h_out

    def forward(self, decoder_input, embedded_inputs, hidden, context, init_mask):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        idxs = None
        mask = Variable(init_mask, requires_grad=False)

        for i in steps:

            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i, context)
            # select the next inputs for the decoder [batch_size x hidden_dim]
            idxs = self.decode(
                probs,
                mask
            )

            idxs = idxs.detach()  # Otherwise pytorch complains it want's a reward, todo implement this more properly?

            # Gather input embedding of selected
            decoder_input = torch.gather(
                embedded_inputs,
                0,
                idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:])
            ).squeeze(0)

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)

        return (torch.stack(outputs, 1), torch.stack(selections, 1))

    def decode(self, probs, mask):
        if self.decode_type == "greedy":
            _, idxs = probs.max(1)
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            idxs = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"

        return idxs

class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits

class PointNet(nn.Module):
    def __init__(self, args={}):
        super(PointNet, self).__init__()

        # network parameters
        self.hidden_size = args['hidden_size']
        self.sort_x_size = args['sort_x_size']

        self.n_glimpses = 0
        self.sort_encoder = LSTMEncoder(embedding_dim=self.hidden_size, hidden_size=self.hidden_size,
                                        num_layers=3, bidirectional=True,
                                        batch_first=True)

        ## For sort_x embedding layer
        self.sort_x_embedding = nn.Linear(in_features=self.sort_x_size, out_features=self.hidden_size, bias=False)

        tanh_clipping = 10
        mask_inner = True
        mask_logits = True
        self.decoder = Decoder(
            self.hidden_size,#self.sort_x_emb_size
            self.hidden_size,#self.sort_emb_size
            tanh_exploration=tanh_clipping,  # tanh_clipping
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


    def get_seq_mask(self, max_seq_len, batch_size, sort_len):
        """
        Get the mask Tensor for sort task
        """
        range_tensor = torch.arange(max_seq_len, device=sort_len.device, dtype=sort_len.dtype).expand(batch_size,
                                                                                                      max_seq_len,
                                                                                                      max_seq_len)
        each_len_tensor = sort_len.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)
        row_mask_tensor = (range_tensor < each_len_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor
        mask_tensor = mask_tensor.bool()
        return mask_tensor

    def get_init_mask(self, max_seq_len, batch_size, sort_len):
        """
        Get the init mask for decoder
        """
        range_tensor = torch.arange(max_seq_len, device=sort_len.device, dtype=sort_len.dtype).expand(batch_size, max_seq_len)
        each_len_tensor = sort_len.view(-1, 1).expand(batch_size, max_seq_len)
        raw_mask_tensor = range_tensor >= each_len_tensor
        return raw_mask_tensor


    def enc_sort_emb(self, sort_emb, sort_len, batch_size, max_seq_len):
        """
        Encode the sort emb and paper the input for Decoder
        """
        sort_encoder_outputs, (sort_encoder_h_n, sort_encoder_c_n) = self.sort_encoder(sort_emb, sort_len, max_seq_len)
        dec_init_state = (sort_encoder_h_n, sort_encoder_c_n)
        decoder_input = sort_encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
        inputs = sort_encoder_outputs.permute(1, 0, 2).contiguous()
        enc_h = sort_encoder_outputs.permute(1, 0, 2).contiguous()  #(seq_len, batch_size, hidden)
        return decoder_input, inputs, dec_init_state, enc_h

    def get_pos(self, idx):
        mask = idx < 0
        max_len = idx.shape[1]
        cols = []
        for v in range(max_len):
            col = torch.zeros_like(idx[:, 0])
            x, y = torch.where(idx == v)
            col[x] = y
            cols.append(col)
        pos = torch.stack(cols, dim=1)
        pos = pos.masked_fill(mask, -1)
        return pos


    '''
    原始样本
    [
        [[2], [1], [3]],
        [[2], [3]],
        [[2], [5], [4], [1]]
    ]
    
    sort_x
    [
        [[2], [1], [3], [0]],
        [[2], [3], [0], [0]],
        [[2], [5], [4], [1]]
    ]
    batch_size * max_len * d
    
    sort_len : batch_size 
    [
        3,
        2,
        4
    ]

    prnet =  PointNet({sort_x_size: 1, hidden_size:16})
    
    pointer_log_scores, pointer_argmaxs, mask_tensor = prnet(sort_len, sort_x)
    
    第一样本-第一个解码步[0.1 0.2 0.7],
    第一样本-第二个解码步[0.9 0.1 0.0],
    第一样本-第三个解码步[1   0  0],
    第一样本-第四个解码步[1   0  0],  xxx
    pointer_log_scores: batch_size * max_len * max_len
    [
        [
            [0.1 0.2 0.7],
            [0.9 0.1 0.0],
            [1   0  0],
            [1   0  0],
        ],
        
        [
         ...
        ]
    ]
    
    
    pointer_argmaxs: batch_size * max_len
    [
        [ 2, 0, 1, ...]
        [ 1, 0, ...]
        [ 1, 2, 0, 3]
    ]
    
    order sort_x, sort_len -> prnet ->  pointer_argmaxs ---> geo mapping -> geo look up --> lstm
    
    '''
    def forward(self, sort_len, sort_x, get_pos=False):
        batch_size = sort_x.size(0)
        max_seq_len = sort_x.size(1)
        init_mask = self.get_init_mask(max_seq_len, batch_size, sort_len)

        sort_x_emb = self.sort_x_embedding(sort_x)  # embedding sort_x, (batch_size, max_seq_len, todo_emb_dim)
        decoder_input, inputs, dec_init_state, enc_h = self.enc_sort_emb(sort_x_emb, sort_len, batch_size, max_seq_len)
        (pointer_log_scores, pointer_argmaxs) = self.decoder(decoder_input, inputs, dec_init_state, enc_h,init_mask)
        mask_tensor = self.get_seq_mask(max_seq_len, batch_size, sort_len)
        pointer_log_scores = pointer_log_scores.exp()
        pos = torch.tensor(0)
        if get_pos:
            # pos = self.get_pos(pointer_argmaxs)
            # pos = pos.masked_fill(init_mask, -1)
            pos = self.get_pos(pointer_argmaxs)
            pos = pos.masked_fill(init_mask, -1)

        return pointer_log_scores, pointer_argmaxs, pos, mask_tensor


