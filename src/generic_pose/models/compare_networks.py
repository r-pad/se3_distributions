# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:58:21 2018

@author: bokorn
"""
import numpy as np

import torch
import torch.nn as nn

from generic_pose.models.symetric_layers import QSymetric, SplitLinear

class CompareNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(CompareNet, self).__init__()

        self.network = nn.Sequential(
                                     nn.Dropout(),
                                     nn.Linear(features_size * 2, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096, int(np.prod(output_dim)))
                                     )
    def forward(self, v1, v2):
        return self.network(torch.cat((v1, v2), dim=1))
        
class SymetricCompareNet(nn.Module):
    def __init__(self, features_size, output_dim = 4):
        super(SymetricCompareNet, self).__init__()
        assert int(np.prod(output_dim)) == 4, 'Only symetric regressioncompare implemented'
        
        self.network = nn.Sequential(
                                     nn.Tanh(),
                                     nn.Dropout(),
                                     SplitLinear(3072, 1024, 3072, 1024),
                                     nn.Tanh(),
                                     SplitLinear(3072, 1024, 3, 1)
                                     )
        self.do = nn.Dropout()
        self.q_symetric = QSymetric(features_size, 3072, 1024)
    
    def forward(self, v1, v2):
        x = self.q_symetric(self.do(v1), self.do(v2))
        return self.network(x)

class SplitCompareNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(SplitCompareNet, self).__init__()
        assert int(np.prod(output_dim)) == 4, 'Only split regressioncompare implemented'

        self.neg_features = 3072
        self.sym_features = 1024
        
        self.neg_network = nn.Sequential(
                                         nn.Tanh(),
                                         nn.Dropout(),
                                         nn.Linear(self.neg_features, 3072, bias=False),
                                         nn.Tanh(),
                                         nn.Linear(3072, 3, bias=False)
                                         )

        self.sym_network = nn.Sequential(
                                         nn.Tanh(),
                                         nn.Dropout(),
                                         nn.Linear(self.sym_features, 1024),
                                         nn.Tanh(),
                                         nn.Linear(1024, 1)
                                         )
        
        self.do = nn.Dropout()
        self.q_symetric = QSymetric(features_size, self.neg_features, self.sym_features)
    
    def forward(self, v1, v2):
        x = self.q_symetric(self.do(v1), self.do(v2))
        x_neg = x[:,:self.neg_features]
        x_sym = x[:,self.neg_features:]
        
        x_neg = self.neg_network(x_neg)
        x_sym = self.sym_network(x_sym)
        
        return torch.cat((x_neg, x_sym), dim=1)

compare_networks = {
                    'basic':CompareNet,
                    'symetric':SymetricCompareNet,
                    'split':SplitCompareNet,
                    }