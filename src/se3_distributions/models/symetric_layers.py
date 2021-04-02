# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:18:19 2018

@author: bokorn
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class QSymetric(nn.Module):
    def __init__(self, in_features, neg_features, sym_features, bias=True):
        super(QSymetric, self).__init__()
        self.in_features = in_features
        self.neg_features = neg_features
        self.sym_features = sym_features
        
        self.neg_weights = Parameter(torch.Tensor(neg_features, in_features))
        self.sym_weights = Parameter(torch.Tensor(sym_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(sym_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.neg_weights.size(1))
        self.neg_weights.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.sym_weights.size(1))
        self.sym_weights.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        neg_output = input1.matmul(self.neg_weights.t()) - input2.matmul(self.neg_weights.t())
        sym_output = input1.matmul(self.sym_weights.t()) + input2.matmul(self.sym_weights.t())
        if self.bias is not None:
            sym_output += self.bias
            
        output = torch.cat((neg_output, sym_output), dim=1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', neg_features=' + str(self.neg_features) \
            + ', sym_features=' + str(self.sym_features) \
            + ', bias=' + str(self.bias is not None) + ')'
        
class SplitLinear(nn.Module):
    def __init__(self, in_neg_features, in_sym_features, out_neg_features, out_sym_features, bias=True):
        super(SplitLinear, self).__init__()
        self.in_neg_features = in_neg_features
        self.in_sym_features = in_sym_features
        self.out_neg_features = out_neg_features
        self.out_sym_features = out_sym_features
        
        self.nn_weights = Parameter(torch.Tensor(out_neg_features, in_neg_features))
        self.ss_weights = Parameter(torch.Tensor(out_sym_features, in_sym_features))
        self.ns_weights = Parameter(torch.Tensor(out_neg_features, in_sym_features))
        self.sn_weights = Parameter(torch.Tensor(out_sym_features, in_neg_features))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_sym_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.nn_weights.size(1))
        self.nn_weights.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.ss_weights.size(1))
        self.ss_weights.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.ns_weights.size(1))
        self.ns_weights.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.sn_weights.size(1))
        self.sn_weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        neg_input = input[:,:self.in_neg_features]
        sym_input = input[:,self.in_neg_features:]
        #nonzeros = torch.nonzero(neg_input)
        sign_input = torch.sign(neg_input[:,0]).unsqueeze(dim=1)

        neg_output = neg_input.matmul(self.nn_weights.t()) + sign_input * sym_input.matmul(self.ns_weights.t())
        sym_output = sign_input * neg_input.matmul(self.sn_weights.t()) + sym_input.matmul(self.ss_weights.t())

        if self.bias is not None:
            sym_output += self.bias
            
        output = torch.cat((neg_output, sym_output), dim=1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_neg_features=' + str(self.in_neg_features) \
            + ', in_sym_features=' + str(self.in_sym_features) \
            + ', out_neg_features=' + str(self.out_neg_features) \
            + ', out_sym_features=' + str(self.out_sym_features) \
            + ', bias=' + str(self.bias is not None) + ')'