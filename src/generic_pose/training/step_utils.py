# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:11:04 2018

@author: bokorn
"""
import numpy as np
import torch
from itertools import repeat
import torch.nn.functional as F
import generic_pose.utils.transformations as tf_trans
from generic_pose.training.utils import to_var, to_np
from generic_pose.utils.data_preprocessing import (resizeAndPad, 
                                                   cropAndResize, 
                                                   transparentOverlay, 
                                                   quatDiff, 
                                                   quatAngularDiff,
                                                   quat2AxisAngle)

from generic_pose.losses.viewpoint_loss import ViewpointLoss, denseViewpointError, dotLoss
class_loss = ViewpointLoss()

def topClass(delta_angle, class_est, use_max = True):
    batch_size = delta_angle.size(0)    
    delta = 0
    if(use_max):
        idx = to_np(torch.max(class_est, 1)[1])
    else:
        idx = to_np(torch.min(class_est, 1)[1])
        
    for j in range(batch_size):
        delta += delta_angle[j, idx[j]]
    return delta/batch_size

def correctTop(delta_angle, class_est, use_max = True):
    if(use_max):
        est_idx = to_np(torch.max(class_est, 1)[1])
    else:
        est_idx = to_np(torch.min(class_est, 1)[1])
    
    true_idx = to_np(torch.min(delta_angle, 1)[1])
    
    return (est_idx==true_idx).mean()

def expClass(delta_angle, class_est, use_softmax = True):
    if(use_softmax):
        prob = to_np(F.softmax(class_est, dim=1))
    else:
        prob = to_np(F.softmin(class_est, dim=1))
        #class_sum = torch.sum(class_est, dim=1)
        #prob = to_np(class_est.div(class_sum.unsqueeze(1).expand_as(class_est)))
    
    return (to_np(delta_angle)*prob).mean()

def getAxes(num_axes):
    axes = []
    if(num_axes in [6, 14, 26]):
        axes.extend([[1,0,0],
                     [0,1,0],
                     [0,0,1],
                     [-1,0,0],
                     [0,-1,0],
                     [0,0,-1]])
    else:
        raise NotImplementedError('{} axes not implemented. Only 6, 14 and 26'.format(num_axes))
    if(num_axes in [14, 26]):
        axes.extend([[1,1,1],
                     [1,1,-1],
                     [1,-1,1],
                     [1,-1,-1],
                     [-1,1,1],
                     [-1,1,-1],
                     [-1,-1,1],
                     [-1,-1,-1]])
    if(num_axes == 26):
        axes.extend([[1,1,0],
                     [1,-1,0],
                     [-1,1,0],
                     [-1,-1,0],
                     [1,0,1],
                     [1,0,-1],
                     [-1,0,1],
                     [-1,0,-1],
                     [0,1,1],
                     [0,1,-1],
                     [0,-1,1],
                     [0,-1,-1]])
    axes = np.array(axes, dtype='float64')
    axes /= np.linalg.norm(axes, axis=1).reshape(-1,1)
    return axes
    

def evaluateStepClass(model, origin_img, query_img, origin_quat, query_quat,
                      bin_axes, step_angle = np.pi/4.0, 
                      use_softmax_labels=True, loss_type = 'dot', 
                      sigma = 0.1,
                      optimizer=None, retain_graph = False, 
                      disp_metrics=False):
    batch_size = origin_quat.size(0)
    num_bins = len(bin_axes) + 1
    delta_angle = torch.zeros(batch_size, num_bins)
    
    if type(step_angle) in [int, float]:
        step_angle = repeat(step_angle)

    for j, (oq, qq, ang) in enumerate(zip(origin_quat, query_quat, step_angle)):
        start_diff = quatAngularDiff(oq, qq)        
        for k, axis in enumerate(bin_axes):
            move_quat = tf_trans.quaternion_about_axis(ang, axis)
            iter_quat = tf_trans.quaternion_multiply(move_quat, oq)
            final_diff = quatAngularDiff(iter_quat, qq)
            delta_angle[j,k] = final_diff - start_diff
    origin_img = to_var(origin_img)
    query_img = to_var(query_img)
    
    results = {}
    if(optimizer is not None):
        optimizer.zero_grad()

    origin_features = model.features(origin_img)
    query_features = model.features(query_img)
    class_est = model.compare_network(origin_features,
                                      query_features)
    if(use_softmax_labels):
        labels = F.softmin(to_var(delta_angle), dim=1)
    else:
        labels = to_var(-delta_angle)

    if(loss_type == 'dot'):
        loss_binned = dotLoss(class_est, labels)
    elif(loss_type == 'sa'):
        labels = torch.exp(labels/sigma)
        loss_binned = class_loss(class_est, labels)
    elif(loss_type == 'mse'):
        loss_binned = F.mse_loss(class_est, -labels)
    elif(loss_type == 'kl'):
        if(not use_softmax_labels):
            raise ValueError('KL Divergence Requires softmin labels')
        loss_binned = F.kl_div(F.log_softmax(class_est), labels)
    else:
        raise ValueError('Invalid Loss Type {}'.format(loss_type))
    if(optimizer is not None):
        loss_binned.backward(retain_graph=retain_graph)
        optimizer.step()
    results['label_vec'] = labels
    results['class_vec'] = class_est
    results['delta_vec'] = delta_angle
    results['loss_{}'.format(loss_type)] = loss_binned.data[0]
    
    if(disp_metrics):
        delta_top = topClass(delta_angle, class_est, use_max = loss_type in ['dot', 'kl', 'sa'])
        delta_exp = expClass(delta_angle, class_est, use_softmax = loss_type in ['dot', 'kl', 'sa'])
        top_correct = correctTop(delta_angle, class_est, use_max = loss_type in ['dot', 'kl', 'sa'])
        results['delta_top'] = delta_top*180.0/np.pi
        results['delta_exp'] = delta_exp*180.0/np.pi
        results['top_correct'] = top_correct
        results['mean_origin_features'] = np.mean(np.abs(to_np(origin_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))        

    return results
    