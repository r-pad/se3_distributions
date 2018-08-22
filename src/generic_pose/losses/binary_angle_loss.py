# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:55:02 2018

@author: bokorn
"""
import numpy as np
import torch

def angle2Sign(angles, target_angle = np.pi/4.0):
    diffs = (angles-target_angle).unsqueeze(1).float()
    return torch.sign(diffs)

def angleLessThan(angles, target_angle = np.pi/4.0):
    return (angles < target_angle).unsqueeze(1).float()
    
def binaryHingeLoss(preds, angles, target_angle = np.pi/4.0):
    diffs = (angles-target_angle).unsqueeze(1).float()
    #labels = torch.sign(diffs)
    #loss = torch.max(0, 1-preds*labels*torch.abs(diffs))
    loss = torch.clamp(1.0-preds*diffs, min=0.0)
    return torch.mean(loss)
    
def binaryAccuracy(preds, angles, target_angle = np.pi/4.0, threshold = 0.5):
    diffs = angles-target_angle 
    #l_sign = torch.sign(diffs)
    #p_sign = torch.sign(preds)
    #zero_one_loss = (l_sign == p_sign).float()
    #angular_error = torch.sum(torch.abs(diffs)*zero_one_loss)/zero_one_loss.sum()
    #accuracy = torch.mean(zero_one_loss)
    #less_accuracy = torch.mean(zero_one_loss*(l_sign < 0).float())
    #greater_accuracy = torch.mean(zero_one_loss*(l_sign > 0).float())
    l_val = angleLessThan(angles, target_angle)
    p_val = (preds > threshold).float()
    zero_one_loss = (l_val == p_val).float()
    angular_error = torch.sum(torch.abs(diffs)*(1.0-zero_one_loss))
    if(((1.0-zero_one_loss).sum() > 0).any()):
        angular_error /= (1.0-zero_one_loss).sum()
    accuracy = torch.mean(zero_one_loss)
    less_accuracy = (zero_one_loss*(l_val)).sum()/l_val.sum()
    greater_accuracy = (zero_one_loss*(1.0-l_val)).sum()/(1.0-l_val).sum()

    return accuracy, less_accuracy, greater_accuracy, angular_error, zero_one_loss
