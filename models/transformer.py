# Transformer model
# This code has been repurposed from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2).float() * (-math.log(10000.0))
        div_term = torch.exp(div_term / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self,src_vocab_size,trg_vocab_size,d_model,nhead,
                 num_encoder_layers,num_decoder_layers,
                 dim_feedforward,dropout,pad_idx,device):
        super(Transformer,self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.activation = 'relu'
        self.device = device

        # Input
        self.src_embedding = nn.Embedding(src_vocab_size,d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size,d_model)
        self.positional_encoding = PositionalEncoding(d_model,dropout)

        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, self.activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, self.activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # Output
        self.linear = nn.Linear(d_model,trg_vocab_size)

        self._reset_parameters()

    def forward(self,src,trg):
        # Masks
        src_mask = None
        trg_mask,src_kp_mask,trg_kp_mask = self._get_masks(src,trg)
        # Input
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        trg = self.trg_embedding(trg)
        trg = self.positional_encoding(trg)
        # Encoder
        memory, enc_attn_wts = self.encoder(src, mask=src_mask,
                                            src_kp_mask=src_kp_mask)
        # Decoder
        memory_mask = None
        memory_kp_mask = None
        out, dec_attn_wts = self.decoder(trg, memory, trg_mask=trg_mask,
                                         memory_mask=memory_mask,
                                         trg_kp_mask=trg_kp_mask,
                                         memory_kp_mask=memory_kp_mask)
        # Output
        out = self.linear(out)
        # Attention weights
        attn_wts = {'Encoder':enc_attn_wts,
                    'Decoder':dec_attn_wts}
        return out, attn_wts

    def _get_masks(self,src,trg):
        sz = trg.shape[0]
        trg_mask = self._generate_square_subsequent_mask(sz)
        trg_mask = trg_mask.to(self.device)
        src_kp_mask = (src == self.pad_idx).transpose(0,1).to(self.device)
        trg_kp_mask = (trg == self.pad_idx).transpose(0,1).to(self.device)
        return trg_mask,src_kp_mask,trg_kp_mask

    def _generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(nn.Module):
    def __init__(self,encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.norm = norm

        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src, mask=None, src_kp_mask=None):
        attn_weights = []
        output = src
        for mod in self.layers:
            output, attn_wts = mod(output, src_mask=mask, src_kp_mask=src_kp_mask)
            attn_weights.append(attn_wts)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.norm = norm

        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, trg, memory, trg_mask=None, memory_mask=None,
                trg_kp_mask=None, memory_kp_mask=None):
        attn_weights = []
        output = trg
        for mod in self.layers:
            output, attn_wts = mod(output, memory, trg_mask=trg_mask,
                memory_mask=memory_mask,
                trg_kp_mask=trg_kp_mask,
                memory_kp_mask=memory_kp_mask)
            attn_weights.append(attn_wts)
        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_kp_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_kp_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation
        self.activation = _get_activation_fn(activation)

    def forward(self, trg, memory, trg_mask=None, memory_mask=None,
                trg_kp_mask=None, memory_kp_mask=None):
        trg2, attn_weights1 = self.self_attn(trg, trg, trg,
                                  attn_mask=trg_mask, key_padding_mask=trg_kp_mask)
        trg = trg + self.dropout1(trg2)
        trg = self.norm1(trg)
        trg2, attn_weights2 = self.multihead_attn(trg, memory, memory, attn_mask=memory_mask,
                                  key_padding_mask=memory_kp_mask)
        trg = trg + self.dropout2(trg2)
        trg = self.norm2(trg)
        trg2 = self.linear2(self.dropout(self.activation(self.linear1(trg))))
        trg = trg + self.dropout3(trg2)
        trg = self.norm3(trg)

        attn_weights = {'Sublayer1' : attn_weights1,
                        'Sublayer2' : attn_weights2}
        return trg, attn_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    msg = "Activation can be relu or gelu, not {}".format(activation)
    raise RuntimeError(msg)
