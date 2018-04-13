# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:57:53 2018

@author: bokorn
"""
import numpy as np

import torch
from torch.autograd import Variable
from generic_pose.losses.quaternion_loss import (quaternionLoss, 
                                                 quaternionError, 
                                                 quaternionAngles,
                                                 axisError,
                                                 clipQuatAngle,
                                                 quaternionInverse,
                                                 quaternionMultiply,
                                                 loopConsistencyLoss,
                                                 axisAngle2Quat)

from generic_pose.training.utils import to_var, to_np

def evaluateMaxedDotQuat(model, origin, query, quat_true, max_angle = np.pi/4.0,
                     optimizer=None, retain_graph = False, 
                     disp_metrics=False):
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

    clipped_est = clipQuatAngle(quat_est, max_angle)
    
    loss_quat = quaternionLoss(clipped_est, quat_true)

    if(optimizer is not None):
        loss_quat.backward(retain_graph=retain_graph)
        optimizer.step()

    results['quat_vec'] = quat_est
    results['loss_quat'] = loss_quat.data[0]
    
    if(disp_metrics):
        ang_errs = quaternionError(clipped_est, quat_true)
        results['errs_vec'] = ang_errs*180.0/np.pi
        ang_diff = quaternionAngles(quat_true)
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['ang_improvement'] = np.mean(ang_diff-ang_errs)*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(origin_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))

    return results
    

def evaluateMaxedTrueQuat(model, origin, query, quat_true, max_angle = np.pi/4.0,
                      optimizer=None, retain_graph = False, 
                      disp_metrics=False):

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

    clipped_true = clipQuatAngle(quat_true.clone(), max_angle)
    loss_quat = quaternionLoss(quat_est, clipped_true)
    if(optimizer is not None):
        loss_quat.backward(retain_graph=retain_graph)
        optimizer.step()

    results['quat_vec'] = quat_est
    results['loss_quat'] = loss_quat.data[0]
    
    if(disp_metrics):
        clipped_est = clipQuatAngle(quat_est, max_angle)
        ang_errs = quaternionError(clipped_est, quat_true)
        results['errs_vec'] = ang_errs*180.0/np.pi
        ang_diff = quaternionAngles(quat_true)
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(origin_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))

    return results    
    
def evaluateMaxedMultQuat(model, im0, im1, q0, q1, max_angle = np.pi/4.0,
                     optimizer=None, retain_graph = False, 
                     disp_metrics=False):
    im0 = to_var(im0)
    im1 = to_var(im1)
    q0 = to_var(q0)
    q1 = to_var(q1)
    q1_inv = quaternionInverse(q1)
    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    f0 = model.features(im0)
    f1 = model.features(im1)
    q_est = model.compare_network(f0, f1)
    clipped_est = clipQuatAngle(q_est, max_angle)
    
    loss_quat, ang_errs = loopConsistencyLoss([clipped_est, q0, q1_inv], calc_angle=True)

    if(optimizer is not None):
        loss_quat.backward(retain_graph=retain_graph)
        optimizer.step()

    results['quat_vec'] = clipped_est
    results['loss_quat'] = loss_quat.data[0]
    
    if(disp_metrics):
        results['errs_vec'] = ang_errs*180.0/np.pi
        quat_true = quaternionMultiply(q0.float(), q1_inv)
        ang_diff = quaternionAngles(quat_true)
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(f0)))
        results['mean_query_features'] = np.mean(np.abs(to_np(f1)))

    return results
    
def evaluateMaxedDotAxisAngle(model, origin, query, quat_true, max_angle = np.pi/4.0,
                     optimizer=None, retain_graph = False, 
                     disp_metrics=False):
    origin = to_var(origin)
    query = to_var(query)
    quat_true = to_var(quat_true)

    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    origin_features = model.features(origin)
    query_features = model.features(query)
    axis_angle_est = model.compare_network(origin_features, 
                                     query_features)

    clipped_est = axisAngle2Quat(axis_angle_est, max_angle)
    
    loss_quat = quaternionLoss(clipped_est, quat_true)

    if(optimizer is not None):
        loss_quat.backward(retain_graph=retain_graph)
        optimizer.step()

    results['quat_vec'] = axis_angle_est
    results['loss_quat'] = loss_quat.data[0]
    
    if(disp_metrics):
        ang_errs = quaternionError(clipped_est, quat_true)
        results['errs_vec'] = ang_errs*180.0/np.pi
        ang_diff = quaternionAngles(quat_true)
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(origin_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))

    return results
    

def evaluateMaxedTrueAxisAngle(model, origin, query, quat_true, max_angle = np.pi/4.0,
                      optimizer=None, retain_graph = False, 
                      disp_metrics=False):

    origin = to_var(origin)
    query = to_var(query)
    quat_true = to_var(quat_true)

    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    origin_features = model.features(origin)
    query_features = model.features(query)
    axis_angle_est = model.compare_network(origin_features, 
                                     query_features)

    quat_est = axisAngle2Quat(axis_angle_est, max_angle=np.inf)
    clipped_true = clipQuatAngle(quat_true.clone(), max_angle)
    loss_quat = quaternionLoss(quat_est, clipped_true)
    if(optimizer is not None):
        loss_quat.backward(retain_graph=retain_graph)
        optimizer.step()

    results['quat_vec'] = quat_est
    results['loss_quat'] = loss_quat.data[0]
    
    if(disp_metrics):
        clipped_est = axisAngle2Quat(axis_angle_est, max_angle)
        ang_errs = quaternionError(clipped_est, quat_true)
        results['errs_vec'] = ang_errs*180.0/np.pi
        ang_diff = quaternionAngles(quat_true)
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(origin_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))

    return results    
    
def evaluateMaxedMultAxisAngle(model, im0, im1, q0, q1, max_angle = np.pi/4.0,
                     optimizer=None, retain_graph = False, 
                     disp_metrics=False):
    im0 = to_var(im0)
    im1 = to_var(im1)
    q0 = to_var(q0)
    q1 = to_var(q1)
    q1_inv = quaternionInverse(q1)
    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    f0 = model.features(im0)
    f1 = model.features(im1)
    axis_angle_est = model.compare_network(f0, f1)
    clipped_est = axisAngle2Quat(axis_angle_est, max_angle)
    
    loss_quat, ang_errs = loopConsistencyLoss([clipped_est, q0, q1_inv], calc_angle=True)

    if(optimizer is not None):
        loss_quat.backward(retain_graph=retain_graph)
        optimizer.step()

    results['quat_vec'] = clipped_est
    results['loss_quat'] = loss_quat.data[0]
    
    if(disp_metrics):
        results['errs_vec'] = ang_errs*180.0/np.pi
        quat_true = quaternionMultiply(q0.float(), q1_inv)
        ang_diff = quaternionAngles(quat_true)
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(f0)))
        results['mean_query_features'] = np.mean(np.abs(to_np(f1)))

    return results