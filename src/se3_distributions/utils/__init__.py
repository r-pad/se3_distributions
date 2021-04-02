# -*- coding: utf-8 -*-
"""
@author: bokorn
"""
import torch
from torch.autograd import Variable

def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x.detach().data.cpu().numpy()

def to_var(x, requires_grad = True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class SingularArray(object):
    def __init__(self, value):
        self.value = value
    def __getitem__(self, index):
        return self.value

