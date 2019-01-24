# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:58:21 2018

@author: bokorn
"""
import numpy as np

import torch
import torch.nn as nn

from generic_pose.models.symetric_layers import QSymetric, SplitLinear
from generic_pose.models.skip_compare import create_skip_compare
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

class SigmoidCompareNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(SigmoidCompareNet, self).__init__()

        self.network = nn.Sequential(
                                     nn.Dropout(),
                                     nn.Linear(features_size * 2, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096, int(np.prod(output_dim))),
                                     nn.Sigmoid()
                                     )
    def forward(self, v1, v2):
        return self.network(torch.cat((v1, v2), dim=1))



class SigmoidWideCompareNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(SigmoidWideCompareNet, self).__init__()

        self.network = nn.Sequential(
                                     nn.Dropout(),
                                     nn.Linear(features_size * 2, 8192),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Linear(8192, 8192),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(8192, int(np.prod(output_dim))),
                                     nn.Sigmoid()
                                     )
    def forward(self, v1, v2):
        return self.network(torch.cat((v1, v2), dim=1))


class SigmoidDeepCompareNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(SigmoidDeepCompareNet, self).__init__()

        self.network = nn.Sequential(
                                     nn.Dropout(),
                                     nn.Linear(features_size * 2, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096, int(np.prod(output_dim))),
                                     nn.Sigmoid()
                                     )
    def forward(self, v1, v2):
        return self.network(torch.cat((v1, v2), dim=1))


class TanhCompareNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(TanhCompareNet, self).__init__()

        self.network = nn.Sequential(
                                     nn.Dropout(),
                                     nn.Linear(features_size * 2, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096, int(np.prod(output_dim))),
                                     nn.Tanh()
                                     )
    def forward(self, v1, v2):
        return self.network(torch.cat((v1, v2), dim=1))


class LinearPreCompareNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(LinearPreCompareNet, self).__init__()

        self.linear = nn.Sequential(
                                    nn.Linear(features_size, 2048),
                                    nn.ReLU(inplace=True)
                                    )


        self.network = nn.Sequential(
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096, int(np.prod(output_dim)))
                                     )
    def forward(self, v1, v2):
        v1 = self.linear(v1)
        v2 = self.linear(v2)
        return self.network(torch.cat((v1, v2), dim=1))
        
class SkipCompareNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(SkipCompareNet, self).__init__()

        self.network = create_skip_compare(inplanes = features_size * 2, 
                                           planes = [4096, 4096],
                                           blocks = [1, 1])
                                           
        self.linear = nn.Linear(4096, int(np.prod(output_dim)))
        
    def forward(self, v1, v2):
        x = self.network(torch.cat((v1, v2), dim=1))
        return self.linear(x)

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

class SingleNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(SingleNet, self).__init__()

        self.network = nn.Sequential(
                                     nn.Dropout(),
                                     nn.Linear(features_size, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096, int(np.prod(output_dim)))
                                     )
    def forward(self, v):
        return self.network(v)


class SigmoidNet(nn.Module):
    def __init__(self, features_size, output_dim):
        super(SigmoidNet, self).__init__()

        self.network = nn.Sequential(
                                     nn.Dropout(),
                                     nn.Linear(features_size, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096, int(np.prod(output_dim))),
                                     nn.Sigmoid()
                                     )
    def forward(self, v):
        return self.network(v)


compare_networks = {
                    'basic':CompareNet,
                    'single':SingleNet,
                    'linear':LinearPreCompareNet,
                    'sigmoid':SigmoidCompareNet,
                    'sigmoid_wide':SigmoidWideCompareNet,
                    'sigmoid_deep':SigmoidDeepCompareNet,
                    'tanh':TanhCompareNet,
                    'skip':SkipCompareNet,
                    'symetric':SymetricCompareNet,
                    'split':SplitCompareNet,
                    }
class_networks = {
        'basic':SingleNet,
        'sigmoid':SigmoidNet,
        }

