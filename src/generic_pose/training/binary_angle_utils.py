# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:46:55 2018

@author: bokorn
"""
import torch

import numpy as np

from generic_pose.losses.quaternion_loss import tensor2Angle
from generic_pose.losses.binary_angle_loss import binaryHingeLoss, binaryAccuracy, angle2Sign, angleLessThan
from generic_pose.training.utils import to_var, to_np

def evaluateBinaryEstimate(model, origin, query, quat, target_angle = np.pi/4.0,
                           prediction_thresh = 0.5, 
                           loss_type='BCEwL',
                           optimizer=None, retain_graph = False, 
                           disp_metrics=False):
    origin = to_var(origin)
    query = to_var(query)
    angles = to_var(tensor2Angle(quat).float())
    
    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    origin_features = model.features(origin)
    query_features = model.features(query)
    preds = model.compare_network(origin_features, 
                                   query_features)

    if(loss_type=='BCEwL'):
        loss =  torch.nn.BCEWithLogitsLoss()
        output = loss(preds, angleLessThan(angles, target_angle=target_angle))
        results['loss_bce_w_ll'] = output.data[0]        
    elif(loss_type=='BCE'):
        loss = torch.nn.BCELoss()
        output = loss(preds, angleLessThan(angles, target_angle=target_angle))
        results['loss_bce'] = output.data[0]
    elif(loss_type=='binaryHinge'):
        output = binaryHingeLoss(preds, angles, target_angle)
        results['loss_hinge'] = output.data[0]
    else:
        raise ValueError('Invalid loss type {}'.format(loss_type))
        
    if(optimizer is not None):
        output.backward(retain_graph=retain_graph)
        optimizer.step()

    
    if(disp_metrics):
        results['angle_vec'] = to_np(angles)*180.0/np.pi
        acc, less_acc, greater_acc, mean_err, zero_one_loss = binaryAccuracy(preds, angles, target_angle, 
                                                                             threshold = prediction_thresh)
        results['output_vec'] = to_np(preds)
        results['correct_vec'] = zero_one_loss
        results['mean_error'] = mean_err.data[0]
        results['accuracy'] = acc.data[0]
        results['accuracy_less'] = less_acc.data[0]
        results['accuracy_greater'] = greater_acc.data[0]

    return results
    
