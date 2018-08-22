# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:10:29 2018

@author: bokorn
"""

import numpy as np

import torch
from generic_pose.losses.quaternion_loss import (quaternionLoss, 
                                                 quaternionError, 
                                                 loopConsistencyLoss,
                                                 quaternionAngles,
                                                 clipQuatAngle,
                                                 blendedLoss,
                                                 axisError)

from generic_pose.losses.viewpoint_loss import ViewpointLoss, denseViewpointError, dotLoss
from generic_pose.utils import to_np, to_var
class_loss = ViewpointLoss()

def evaluatePairReg(model, origin, query, quat_true,
                    optimizer=None, retain_graph = False, 
                    disp_metrics=False, threshold = 5*np.pi/180, clip = None):

    origin = to_var(origin)
    query = to_var(query)
    quat_true = to_var(quat_true)

    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    origin_features = model.features(origin)
    query_features = model.features(query)
    quat_est = model.compare_network(origin_features, 
                                     query_features)

    loss_quat = quaternionLoss(quat_est, quat_true)

    if(optimizer is not None):
        loss_quat.backward(retain_graph=retain_graph)
        if(clip is not None):
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

    results['quat_vec'] = quat_est
    results['loss_quat'] = loss_quat.data[0]
    
    if(disp_metrics):
        ang_errs = quaternionError(quat_est, quat_true)
        ang_diff = quaternionAngles(quat_true)
        results['errs_vec'] = ang_errs*180.0/np.pi
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['thresh_{}'.format(int(threshold*180/np.pi))] = np.mean(ang_errs<threshold)
        results['mean_origin_features'] = np.mean(np.abs(to_np(origin_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))

    return results

def evaluatePairBlendReg(model, origin, query, quat_true, 
                         min_angle = np.pi/4.0, max_angle = np.pi,
                         optimizer=None, retain_graph = False, 
                         disp_metrics=False, threshold = 5*np.pi/180, clip = None):

    origin = to_var(origin)
    query = to_var(query)
    quat_true = to_var(quat_true)

    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    origin_features = model.features(origin)
    query_features = model.features(query)
    quat_est = model.compare_network(origin_features, 
                                     query_features)

    loss_quat = blendedLoss(quat_est, quat_true,
                            min_angle=min_angle,
                            max_angle=max_angle)

    if(optimizer is not None):
        loss_quat.backward(retain_graph=retain_graph)
        if(clip is not None):
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

    results['quat_vec'] = quat_est
    results['loss_quat'] = loss_quat.data[0]
    
    if(disp_metrics):
        ang_errs = quaternionError(quat_est, quat_true)
        ang_diff = quaternionAngles(quat_true)
        axis_err = axisError(quat_est, quat_true)
        results['errs_vec'] = ang_errs*180.0/np.pi
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['axis_err'] = np.mean(axis_err)*180.0/np.pi
        results['thresh_{}'.format(int(threshold*180/np.pi))] = np.mean(ang_errs<threshold)
        results['mean_origin_features'] = np.mean(np.abs(to_np(origin_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))

    return results

def evaluatePairCls(model, origin, query, class_true, num_bins,
                    optimizer=None, retain_graph = False, disp_metrics=False,
                    clip = None):

    origin = to_var(origin)
    query = to_var(query)
    class_true = to_var(class_true)

    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    origin_features = model.features(origin)
    query_features = model.features(query)
    class_est = model.compare_network(origin_features,
                                      query_features)
    
    loss_binned = dotLoss(class_est, class_true)
    
    if(optimizer is not None):
        loss_binned.backward(retain_graph=retain_graph)
        if(clip is not None):
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        
    results['class_vec'] = class_est
    results['loss_binned'] = loss_binned.data[0]
    
    if(disp_metrics):
        err_binned, err_idx = denseViewpointError(class_est, class_true, num_bins)
        results['err_binned'] = err_binned*180.0/np.pi
        results['err_idx'] = err_idx
        results['mean_origin_features'] = np.mean(np.abs(to_np(origin_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))        

    return results

def evaluateLoopReg(model, images, trans_true, loop_truth,
                    optimizer=None, retain_graph = False, disp_metrics=False, 
                    clip=None):
    
    assert len(loop_truth) == len(images), 'Loop length must match loop truth'

    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    loop_transforms = [] 
    loop_len = len(loop_truth)
    
    for j, truth in enumerate(loop_truth):
        if(truth):
            loop_transforms.append(to_var(trans_true[j]))
        else:
            loop_transforms.append(model.forward(to_var(images[j]), to_var(images[(j+1)%loop_len])))
    
    loss_loop, angle_loop = loopConsistencyLoss(loop_transforms, calc_angle=disp_metrics)
        
    if(optimizer is not None):
        loss_loop.backward(retain_graph=retain_graph)
        if(clip is not None):
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
    
    results['loss_loop'] = loss_loop.data[0]
    if(disp_metrics):
        results['err_loop'] = angle_loop*180.0/np.pi
    
    return results
