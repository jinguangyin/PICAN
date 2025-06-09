# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:48:39 2024

@author: HP
"""

import numpy as np
import os

class DataLoader(object):
    def __init__(self, x, t, y, batch_size):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = len(x)
        self.num_batch = int(self.size // self.batch_size)
        self.x = x
        self.y = y
        self.t = t

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        x, t, y = self.x[permutation], self.t[permutation], self.y[permutation]
        self.x = x
        self.y = y
        self.t = t

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.x[start_ind: end_ind, ...]
                y_i = self.y[start_ind: end_ind, ...]
                t_i = self.t[start_ind: end_ind, ...]
                yield (x_i, t_i, y_i)
                self.current_ind += 1
        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

    
def load_dataset(x, t_index, t_diff_data, user_data, y, batch_size):
    data = {}
    x_train = x[0:int(len(x)*0.7)]
    x_val = x[int(len(x)*0.7):int(len(x)*0.85)]
    x_test = x[int(len(x)*0.85):]
    y_train = y[0:int(len(y)*0.7)]
    y_val = y[int(len(y)*0.7):int(len(y)*0.85)]
    y_test = y[int(len(y)*0.85):]   
    t_train = t_index[0:int(len(x)*0.7)]
    t_val = t_index[int(len(x)*0.7):int(len(x)*0.85)]
    t_test = t_index[int(len(x)*0.85):]    
    t_diff_train = t_diff_data[0:int(len(x)*0.7)]
    t_diff_val = t_diff_data[int(len(x)*0.7):int(len(x)*0.85)]
    t_diff_test = t_diff_data[int(len(x)*0.85):]    
    user_train = user_data[0:int(len(x)*0.7)]
    user_val = user_data[int(len(x)*0.7):int(len(x)*0.85)]
    user_test = user_data[int(len(x)*0.85):]     
    data['train_loader'] = DataLoader(x_train, t_train, t_diff_train, user_train, y_train, batch_size)
    data['val_loader'] = DataLoader(x_val, t_val, t_diff_val, user_val, y_val, batch_size)
    data['test_loader'] = DataLoader(x_test, t_test, t_diff_test, user_test, y_test, batch_size)
    return data