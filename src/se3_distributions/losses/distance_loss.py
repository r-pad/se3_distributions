# -*- coding: utf-8 -*-
"""
@author: bokorn
"""

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from se3_distributions.losses.quaternion_loss import tensor2Angle
from se3_distributions.utils import to_np

def rawDistanceLoss(pred, labels, falloff_angle=np.pi/4, reduction='elementwise_mean'):
    if(len(labels.shape) == 2):
        theta = tensor2Angle(labels).float()/np.pi 
    else:
        theta = labels.float()
    loss = F.mse_loss(pred, theta, reduction=reduction)
    return loss

def rawDistanceError(pred, labels, falloff_angle, mean = False):
    if(len(labels.shape) == 2):
        theta = tensor2Angle(labels)
    else:
        theta = labels
    theta_true = to_np(theta)
    theta_pred = to_np(pred) * np.pi
    diff = np.abs(theta_true - theta_pred)
    if(mean):
        return np.mean(diff)
    else:
        return diff

def logFalloffTheta(labels, falloff_angle):
    if(len(labels.shape) == 2):
        theta = tensor2Angle(labels) 
    else:
        theta = labels
    
    log_theta = torch.log(theta/falloff_angle)
    target = theta.clamp(max=1)+(log_theta).clamp(min=0)
    max_target = 1 + np.log(np.pi/falloff_angle)
    return target / max_target


def logDistanceLoss(pred, labels, falloff_angle=np.pi/4, reduction='elementwise_mean'):
    target = logFalloffTheta(labels.float(), falloff_angle) 
    loss = F.mse_loss(pred, target, reduction=reduction)
    return loss

def logDistanceError(pred, labels, falloff_angle, mean = False):
    if(len(labels.shape) == 2):
        theta = tensor2Angle(labels)
    else:
        theta = labels
     
    theta_true = to_np(theta)
    max_target = 1 + np.log(np.pi/falloff_angle)
    denorm_pred = to_np(pred) * max_target
    theta_pred = np.clip((denorm_pred * falloff_angle) * (denorm_pred < 1) \
                         + (np.exp(denorm_pred - 1) * falloff_angle) * (denorm_pred >= 1),
                         0, np.pi) 
    diff = np.abs(theta_true - theta_pred)
    if(mean):
        return np.mean(diff)
    else:
        return diff

def expDecayTheta(labels, falloff_angle):
    if(len(labels.shape) == 2):
        theta = tensor2Angle(labels) 
    else:
        theta = labels

    return torch.exp(-theta/falloff_angle)

def invExpDecayTheta(pred, falloff_angle):
    return np.clip(-np.log(np.maximum(pred, 0))*falloff_angle, 0, np.pi)

def expDistanceLoss(pred, labels, falloff_angle=np.pi/9, reduction='elementwise_mean'):
    target = expDecayTheta(labels.float(), falloff_angle)
    loss = F.mse_loss(pred, target, reduction=reduction)
    return loss

def expDistanceError(pred, labels, falloff_angle, mean = False):
    if(len(labels.shape) == 2):
        theta = tensor2Angle(labels) 
    else:
        theta = labels

    theta_true = to_np(theta)
    theta_pred = invExpDecayTheta(to_np(pred), falloff_angle)
    diff = np.abs(theta_true - theta_pred)
    if(mean):
        return np.mean(diff)
    else:
        return diff

def negExpTheta(labels, falloff_angle):
    if(len(labels.shape) == 2):
        theta = tensor2Angle(labels) 
    else:
        theta = labels

    return 1-torch.exp(-theta/falloff_angle)

def negExpDistanceLoss(pred, labels, falloff_angle=np.pi/9, reduction='elementwise_mean'):
    target = negExpTheta(labels.float(), falloff_angle)
    loss = F.mse_loss(pred, target, reduction=reduction)
    return loss

def negExpDistanceError(pred, labels, falloff_angle, mean = False):
    if(len(labels.shape) == 2):
        theta = tensor2Angle(labels) 
    else:
        theta = labels

    theta_true = to_np(theta)
    theta_pred = np.clip(-np.log(np.maximum(1-to_np(pred), 0))*falloff_angle, 0, np.pi)
    diff = np.abs(theta_true - theta_pred)
    if(mean):
        return np.mean(diff)
    else:
        return diff


