# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn 
"""

import torch
import numpy as np

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.sizes = np.array([len(d) for d in self.datasets])
        self.cumsize = np.cumsum(self.sizes)

    def __getitem__(self, index):
        cum_index = np.nonzero(self.cumsize <= index)[0]
        if(len(cum_index) == 0):
            dataset_index = 0
            sub_index = index
        else:
            sub_index = index - self.cumsize[cum_index[0]]
            dataset_index = cum_index[0] + 1
            
        return self.datasets[dataset_index][sub_index]

    def __len__(self):
        return sum(self.sizes)

