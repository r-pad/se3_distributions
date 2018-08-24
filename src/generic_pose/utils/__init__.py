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
        return x.data.cpu().numpy()

def to_var(x, requires_grad = True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

