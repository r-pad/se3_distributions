# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 21:31:38 2018

@author: bokorn
"""
import torch
from torch.nn.functional import l1_loss

def featureLoss(features_true, features_est, fix_truth = False):
    if(fix_truth):
        features_true = features_true.detach()
    loss = l1_loss(features_est, features_true)
    return loss