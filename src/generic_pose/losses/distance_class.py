# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn 
"""
import numpy as np

import torch
import torch.nn.functional as F

from generic_pose.utils import to_var, to_np
from generic_pose.eval.posecnn_eval import evaluateQuat, getYCBThresholds
from generic_pose.utils.pose_processing import tensorAngularDiff, tensorAngularAllDiffs, getGaussianKernal
from enum import Enum
class LossType(Enum):
    CrossEntropy = 1
    MSE = 2

def vertexDistanceLoss(pred_cls, label_qs, verts, falloff_angle = np.pi/9.0,
                       loss_type = LossType.CrossEntropy):
    labels = to_var(torch.exp(-tensorAngularAllDiffs(label_qs,verts)/falloff_angle)).float()
    if(loss_type == LossType.CrossEntropy):
        loss = -(labels * F.log_softmax(pred_cls, dim=1)).mean()
    elif(loss_type == LossType.MSE):
        loss = F.mse_loss(pred_cls, labels, reduction='mean')
    else:
        raise TypeError('Invalid Lose Type: {}'.format(loss_type))

    return loss

def vertexDistanceMetrics(pred_cls, label_qs, verts, add_data = None, kernal = None):
    with torch.no_grad():
        if(kernal is not None):
            pred_cls = torch.mm(pred_cls, kernal)
        
        metrics = {}
        top_idxs = torch.argmax(pred_cls, dim=1)
        top_qs = verts[top_idxs, :]
        pred_sort = torch.argsort(pred_cls, dim=1, descending=True)
        
        true_dists = tensorAngularAllDiffs(label_qs,verts)
        true_sort = torch.argsort(true_dists, dim=1, descending=False)
        true_idxs = torch.argmin(tensorAngularAllDiffs(label_qs,verts), dim=1)
        
        metrics['top_dist'] = tensorAngularDiff(label_qs, top_qs)*180/np.pi
        metrics['gt_rank'] = torch.nonzero(pred_sort == true_idxs.unsqueeze(1))[:,1]
        metrics['top_rank'] = torch.nonzero(true_sort == top_idxs.unsqueeze(1))[:,1] 
        
        if(add_data is not None):
            metrics['add_error'] = []
            metrics['add_accuracy'] = []
            points = add_data['points']
            use_sym = add_data['use_sym']
            for q_true, q_pred, t_true, t_pred in zip(label_qs, top_qs, 
                add_data['trans_true'], add_data['trans_pcnn']):
                if(t_pred is not None):
                    add_err = evaluateQuat(to_np(q_true), to_np(q_pred), points, use_sym=use_sym, 
                                           t_pred = t_pred, t_true = t_true)
                    metrics['add_error'].append(add_err)
                    metrics['add_accuracy'].append(add_err < add_data['threshold'])
                #else:
                    #pass
                    #metrics['add_error'].append(float('NAN'))
    return metrics

def tripletLoss(pred_cls, label_qs, verts):
    diff = tensorAngularDiff(label_qs, verts)
