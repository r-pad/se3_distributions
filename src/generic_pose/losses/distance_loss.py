# -*- coding: utf-8 -*-
"""
@author: bokorn
"""

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from generic_pose.losses.quaternion_loss import tensor2Angle
from generic_pose.utils import to_np

def rawDistanceLoss(pred, labels, falloff_angle=np.pi/4, reduce=True):
    target = tensor2Angle(labels).float()/np.pi 
    loss = F.mse_loss(pred, target, reduce=reduce)
    return loss

def rawDistanceError(pred, labels, falloff_angle, mean = False):
    theta_true = to_np(tensor2Angle(labels))
    theta_pred = to_np(pred) * np.pi
    diff = np.abs(theta_true - theta_pred)
    if(mean):
        return np.mean(diff)
    else:
        return diff

def logFalloffTheta(labels, falloff_angle):
    theta = tensor2Angle(labels)/falloff_angle
    log_theta = torch.log(theta)
    target = theta.clamp(max=1)+(log_theta).clamp(min=0)
    max_target = 1 + np.log(np.pi/falloff_angle)
    return target / max_target


def logDistanceLoss(pred, labels, falloff_angle=np.pi/4, reduce=True):
    target = logFalloffTheta(labels.float(), falloff_angle) 
    loss = F.mse_loss(pred, target, reduce=reduce)
    return loss

def logDistanceError(pred, labels, falloff_angle, mean = False):
    theta_true = to_np(tensor2Angle(labels))
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
    theta = tensor2Angle(labels)/falloff_angle
    return torch.exp(-theta)

def expDistanceLoss(pred, labels, falloff_angle=np.pi/9, reduce=True):
    target = expDecayTheta(labels.float(), falloff_angle)
    loss = F.mse_loss(pred, target, reduce=reduce)
    return loss

def expDistanceError(pred, labels, falloff_angle, mean = False):
    theta_true = to_np(tensor2Angle(labels))
    theta_pred = np.clip(-np.log(np.maximum(to_np(pred), 0))*falloff_angle, 0, np.pi)
    diff = np.abs(theta_true - theta_pred)
    if(mean):
        return np.mean(diff)
    else:
        return diff

def negExpTheta(labels, falloff_angle):
    theta = tensor2Angle(labels)/falloff_angle
    return 1-torch.exp(-theta)

def negExpDistanceLoss(pred, labels, falloff_angle=np.pi/9, reduce=True):
    target = negExpTheta(labels.float(), falloff_angle)
    loss = F.mse_loss(pred, target, reduce=reduce)
    return loss

def negExpDistanceError(pred, labels, falloff_angle, mean = False):
    theta_true = to_np(tensor2Angle(labels))
    theta_pred = np.clip(-np.log(np.maximum(1-to_np(pred), 0))*falloff_angle, 0, np.pi)
    diff = np.abs(theta_true - theta_pred)
    if(mean):
        return np.mean(diff)
    else:
        return diff


