# Transformer model
# This code has been repurposed from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import torch
import torch.nn as nn
from torch.nn import Transformer

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
        self.device = device

        self.src_embedding = nn.Embedding(src_vocab_size,d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size,d_model)
        self.positional_encoding = PositionalEncoding(d_model,dropout)
        self.transformer = nn.Transformer(d_model,nhead,num_encoder_layers,
                                          num_decoder_layers,dim_feedforward,
                                          dropout)
        self.linear = nn.Linear(d_model,trg_vocab_size)

    def forward(self,src,trg):
        trg_mask,src_kp_mask,trg_kp_mask = self.get_masks(src,trg)
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        trg = self.trg_embedding(trg)
        trg = self.positional_encoding(trg)
        out = self.transformer(src,trg,tgt_mask=trg_mask,
                               src_key_padding_mask=src_kp_mask,
                               tgt_key_padding_mask=trg_kp_mask)
        out = self.linear(out)
        return out


    def get_masks(self,src,trg):
        sz = trg.shape[0]
        trg_mask = self.transformer.generate_square_subsequent_mask(sz)
        src_kp_mask = (src == self.pad_idx).transpose(0,1).to(self.device)
        trg_kp_mask = (trg == self.pad_idx).transpose(0,1).to(self.device)
        return trg_mask,src_kp_mask,trg_kp_mask
