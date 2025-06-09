# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:06:08 2024

@author: HP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, hidden):
        super(PositionalEncoding, self).__init__()
        self.fc = nn.Linear(1, hidden, bias=False)
        self.hidden = hidden
        
    def forward(self, time_diff):
        max_len = time_diff.size(1)
        out = torch.zeros((1, max_len, self.hidden))
        x = self.fc(time_diff)
        out[:, :, 0::2] = torch.sin(x)
        out[:, :, 1::2] = torch.cos(x)
        return out


class SelfAttention(nn.Module):

    def __init__(self, dim_in, dim_k):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_k, bias=False)
        self._norm_fact = np.sqrt(dim_k)
        self.pos_enc = PositionalEncoding(dim_k)

    def forward(self, time_diff, user_emb):
        # x: batch, n, dim_in
        x = time_diff + user_emb
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) /self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)
        out = torch.sum(att, dim=1)
        return out


class wnet(nn.Module):
    def __init__(self, dropout=0.3, in_dim=1, out_dim=1,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2,
                  n_clusters=10, n_z=256):
        super(wnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv1d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=1)
        
        self.fc_a = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=1,
                                    kernel_size=1)
        self.fc_b = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=1,
                                    kernel_size=1)
        self.fc_r = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=1,
                                    kernel_size=1)   
        self.fc_n = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=1,
                                    kernel_size=1)       
        
        self.day_emb = nn.Embedding(7, residual_channels)
        self.hour_emb = nn.Embedding(24, residual_channels)
        self.cas_emb = SelfAttention(residual_channels,residual_channels-1)
                
        # self.day_emb = nn.Parameter(torch.randn(7, residual_channels))
        # self.hour_emb = nn.Parameter(torch.randn(24, residual_channels))
         
        receptive_field = 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size, dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1))
                self.bn.append(nn.BatchNorm1d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=1,
                                    bias=True)

        self.receptive_field = receptive_field
        # self.gate_fusion = GatedFusion(skip_channels, in_dim)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z))
        self.cluster_fc = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=n_z,
                                  kernel_size=1,
                                  bias=True)        
        self.v = 1
        self.bn_emb = nn.BatchNorm1d(skip_channels)
        nn.init.xavier_normal_(self.cluster_layer.data)

    def compute_q(self, z, c):
        z = z.squeeze(2)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()   
        return q

    def forward(self, input, t, time_index, time_diff, user_emb):
        in_len = input.size(2)
        cas_list = []
        for i in range(time_diff.size(1)):
            c_emb = self.cas_emb(time_diff[:,i], user_emb[:,i])
            cas_list.append(c_emb)
        total_c_emb = torch.stack(cas_list, dim=2)
        x = torch.cat((input, total_c_emb),dim=1)
        
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)

        hour_emb = self.hour_emb(time_index[:,0].long()).unsqueeze(2)
        day_emb = self.day_emb(time_index[:,1].long()).unsqueeze(2)
        x = x + hour_emb + day_emb 
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :,  -s.size(2):]
            except:
                skip = 0
            skip = s + skip
            x = self.residual_convs[i](x)
            x = x + residual[:, :, -x.size(2):]

            x = self.bn[i](x)

        # x = F.relu(skip) #(b, 256, 1)
        x = skip
        # z = self.cluster_fc(F.relu(x))
        q = self.compute_q(x, self.cluster_layer)
        cluster_label = torch.argmax(q, dim=1)
        cluster_emb = self.cluster_layer[cluster_label].unsqueeze(2)
        # x = torch.cat([x, cluster_emb], dim=1)
        x = x + cluster_emb
        x = self.bn_emb(x)
        delta = F.relu(self.end_conv_1(x))
        delta = self.end_conv_2(delta)         
        # print(delta.shape)
        p_a = F.softplus(self.fc_a(x))
        p_b = self.fc_b(x)
        p_r = F.softplus(self.fc_r(x))
        p_n = F.softplus(self.fc_n(x))
        out = p_a - p_n*torch.log(1+torch.exp(p_b - p_r*t))
        return out, delta, p_a, p_b, p_r, p_n, q, skip 

