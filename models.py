#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span


class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    """

    def __init__(self, hidden_size,
                 dropout, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size  # size of a single head
        # self.attn_span = attn_span
        # self.enable_mem = enable_mem
        # self.adapt_span_enabled = adapt_span_params['adapt_span_enabled']
        # if self.adapt_span_enabled and self.enable_mem:
        #     self.adaptive_span = AdaptiveSpan(attn_span=attn_span,
        #                                       **adapt_span_params, **kargs)

    def forward(self, query, key, value):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H
        #
        # if self.adapt_span_enabled:
        #     # [optional] trim out memory to reduce unnecessary computation
        #     key, value, key_pe = self.adaptive_span.trim_memory(
        #         query, key, value)

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn = torch.matmul(query, key.transpose(-1, -2))

        attn = attn / math.sqrt(self.hidden_size)  # B x M X (M+L)
        attn = F.softmax(attn, dim=-1)

        # if self.adapt_span_enabled and self.enable_mem:
        #     # trim attention lengths according to the learned span
        #     attn = self.adaptive_span(attn)

        attn = self.dropout(attn)  # B x M X (M+L)

        out = torch.matmul(attn, value)  # B x M x H

        return out

    # def get_cache_size(self):
    #     if self.adapt_span_enabled:
    #         return self.adaptive_span.get_cache_size()
    #     else:
    #         return self.attn_span


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

        # note that the linear layer initialization in current Pytorch is kaiming uniform init

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x


    def forward(self, query, key, value):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d,
            'layer': nn.LayerNorm
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):
        # xavier_uniform initialization
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):


            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class TransformerSeqLayer(nn.Module):
    def __init__(self, hidden_size,  normalization, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(
            hidden_size=hidden_size,  **kargs)
        self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        self.norm1 = Normalization(hidden_size, normalization)
        self.norm2 = Normalization(hidden_size, normalization)

        # self.enable_mem = enable_mem

    def forward(self, h, h_all):
        # h = B x M x H
        # h_cache = B x L x H
        # if self.enable_mem:
        #     h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        # else:
        #     h_all = h_cache                         # B x M x H
        attn_out = self.attn(h, h_all, h_all)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out


class EncoderSeq(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, encoder_nb_layers,
                  **kargs):
        nn.Module.__init__(self)
        # init embeddings
        self.init_embed = nn.Linear(state_size, hidden_size)

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(
                hidden_size=hidden_size, nb_heads=nb_heads,
                 **kargs)
            for _ in range(encoder_nb_layers))

    def forward(self, x):
        # x size = B x M
        # block_size = x.size(1)
        h = self.init_embed(x)  # B x M x H
        # h_cache_next = []
        for l, layer in enumerate(self.layers):
            # cache_size = layer.attn.attn.get_cache_size()

            # B x L x H
            # h_cache_next_l = torch.cat(
            #     [h_cache[l][:, -cache_size + 1:, :], h[:, 0:1, :]],
            #     dim=1).detach()
            #
            # h_cache_next.append(h_cache_next_l)

            h = layer(h, h)  # B x M x H

        return h


class QDecoder(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, decoder_nb_layers,
                 **kargs):
        nn.Module.__init__(self)
        # init embeddings
        self.init_embed = nn.Linear(state_size, hidden_size)

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(
                hidden_size=hidden_size,  nb_heads=nb_heads
                , **kargs)
            for _ in range(decoder_nb_layers))

    def forward(self, x, embedding):
        # x size = B x Q_M
        block_size = x.size(1)
        h = self.init_embed(x)  # B x Q_M x H
        # h_cache_next = []
        for l, layer in enumerate(self.layers):

            h = layer(h, embedding)  # B x Q_M x H

        return h
