# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:21:43 2025

@author: HP
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
from pican_model import wnet
import numpy as np

pre_len = 192//2
twitter_data = np.load(r'twitter_data/day_pad_list.npy')
obs_len = 12//2
x_data =  twitter_data[:,:,0]
y_data =  twitter_data[:,-1,0]
y_data = twitter_data[:,-1,0]-twitter_data[:,0:obs_len,0][:,-1]
pro_data = np.load(r'twitter_data/pro_time_full.npy')
t_data = pro_data[:,1:3]
t_diff_data = np.load(r'twitter_data/diff_time.npy')
user_data = np.load(r'twitter_data/user_feature.npy')

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
class MinmaxScaler():
    def __init__(self, max_data, min_data):
        self.max = max_data
        self.min = min_data

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

scaler = StandardScaler(mean = twitter_data[0:int(len(twitter_data)*0.8),:,0].mean(), 
                        std = twitter_data[0:int(len(twitter_data)*0.8),:,0].std())
x_scale = scaler.transform(x_data)
y_scale = scaler.transform(y_data)
dataloader = utils.load_dataset(x_data, t_data, t_diff_data, user_data, y_data, 64)

net = wnet(dropout=0.3, in_dim=1, out_dim=1,
           residual_channels=32, dilation_channels=32, skip_channels=256,
           end_channels=512, kernel_size=2, blocks=6, layers=2,
           n_clusters=5, n_z=256).cuda()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

title = 'twitter'
best_mae = 10000
early_flag = 0
for epoch in range(100):
    mae_list = []
    mape_list = []
    dataloader['train_loader'].shuffle() 
    for iter, (x, t_e, y) in enumerate(dataloader['train_loader'].get_iterator()):
        net.train()
        train_x = torch.log(torch.Tensor(x[:,0:obs_len]).unsqueeze(1)).cuda()
        label_x = torch.log(torch.Tensor(x)).cuda()
        train_y = torch.log(torch.Tensor(y).unsqueeze(1)).cuda()
        train_t = torch.Tensor(t_e).cuda()
        train_t_diff = torch.Tensor(t_diff_data).cuda()
        train_user = torch.Tensor(user_data).cuda()
        t = torch.arange(0,pre_len).unsqueeze(1).cuda()
        pred, delta, _, _, _, _ , q, _ = net(train_x, t, train_t, train_t_diff, train_user)
        p = target_distribution(q.data)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')        
        pred_loss = torch.mean(torch.abs(delta.squeeze(2) - train_y))
        pinn_loss = torch.mean(torch.abs(torch.log(torch.exp(delta)+torch.exp(train_x[:,:,-1]))-pred[:,-1,:]))
        reconstruction_loss = torch.mean(torch.abs(pred.squeeze(2) - label_x))
        loss = pred_loss + pinn_loss + kl_loss + reconstruction_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)
        optimizer.step()  
    for iter, (x, t_e, y) in enumerate(dataloader['val_loader'].get_iterator()):
        val_x = torch.log(torch.Tensor(x[:,0:obs_len]).unsqueeze(1)).cuda()
        val_y = torch.log(torch.Tensor(y).unsqueeze(1)).cuda()
        val_t = torch.Tensor(t_e).cuda()
        val_t_diff = torch.Tensor(t_diff_data).cuda()
        val_user = torch.Tensor(user_data).cuda()        
        t = torch.arange(0,pre_len).unsqueeze(1).cuda()
        net.eval()
        with torch.no_grad():
            pred, delta, _, _, _, _, q, _ = net(val_x, t, val_t, val_t_diff, val_user)   
            l1_loss = torch.mean(torch.abs(delta.squeeze(2) - val_y))   
            mape_loss = torch.mean(torch.abs(delta.squeeze(2) - val_y)/(val_y+1))
        mae_list.append(l1_loss.data.cpu().numpy())
        mape_list.append(mape_loss.cpu().numpy())
    print('epoch=', epoch, 'mae_loss=', np.mean(mae_list))
    if np.mean(mae_list) < best_mae:
        torch.save(net.state_dict(),'pinn_model_'+ title +'.pkl')
        early_flag = 0
        best_mae = np.mean(mae_list)
    else:
        early_flag = early_flag + 1
    if early_flag > 50: 
        break    
