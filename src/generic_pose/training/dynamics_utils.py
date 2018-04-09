# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 21:54:43 2018

@author: bokorn
"""

import numpy as np

import torch
from torch.nn.functional import l1_loss

from generic_pose.training.utils import to_var, to_np
from generic_pose.losses.quaternion_loss import (quaternionLoss,
                                                 quaternionError,
                                                 quaternionAngles)

def evaluateFeatureDynamics(model, img0, img1, quat01,
                            optimizer=None, retain_graph = False, 
                            disp_metrics=False, fix_truth = False,
                            gamma = 0.1):
    img0 = to_var(img0)
    img1 = to_var(img1)
    quat10 = quat01.clone()
    quat10[:,:3] *= -1.0
    quat01 = to_var(quat01).float()
    quat10 = to_var(quat10).float()
    

    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    feat0 = model.features(img0)
    feat1 = model.features(img1)
    
    quat01_est = model.compare_network(feat0, feat1)
    quat10_est = model.compare_network(feat1, feat0)

    quat_true = torch.cat((quat01, quat10), 0)
    quat_est = torch.cat((quat01_est, quat10_est), 0)

    loss_quat = quaternionLoss(quat_est, quat_true)
    #loss_quat01 = quaternionLoss(quat01_est, quat01)
    #loss_quat10 = quaternionLoss(quat10_est, quat10)
  
    feat1_est = model.dynamics(feat0, quat01)
    feat0_est = model.dynamics(feat1, quat10)
    
    feat_true = torch.cat((feat0, feat1), 0)
    feat_est = torch.cat((feat0_est, feat1_est), 0)
    
    if(fix_truth):
        loss_feat = l1_loss(feat_est, feat_true.detach())
#        loss_feat0 = l1_loss(feat0_est, feat0.detach())
#        loss_feat1 = l1_loss(feat1_est, feat1.detach())
    else:
        loss_feat = torch.mean(torch.abs(feat_est - feat_true))
#        loss_feat0 = l1_loss(feat0_est, feat0)
#        loss_feat1 = l1_loss(feat1_est, feat1)

    loss_joint = loss_quat + gamma*loss_feat
    if(optimizer is not None):
        loss_joint.backward(retain_graph=True)
        #loss_quat01.backward(retain_graph=True)
        #loss_quat10.backward(retain_graph=True)
        #loss_feat.backward(retain_graph=retain_graph)
        #loss_feat0.backward(retain_graph=True)
        #loss_feat1.backward(retain_graph=retain_graph)
        optimizer.step()
    
    results['quat_vec'] = quat_est
    results['loss_quat'] = loss_quat.data[0]
    results['loss_feat'] = loss_feat.data[0]
    results['loss_joint'] = loss_joint.data[0]
    
    if(disp_metrics):
        ang_errs = quaternionError(quat_est, quat_true)
        ang_diff = quaternionAngles(quat_true)
        results['errs_vec'] = ang_errs*180.0/np.pi
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(feat0)))
        results['mean_query_features'] = np.mean(np.abs(to_np(feat1)))    
    
    return results