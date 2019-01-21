# -*- coding: utf-8 -*-
"""
@author: bokorn
"""
import time
import numpy as np
from functools import partial
import torch
import itertools

from quat_math import invAngularPDF

from generic_pose.utils import to_var, to_np
from generic_pose.losses.distance_loss import (rawDistanceLoss,
                                               rawDistanceError,
                                               logDistanceLoss,
                                               logDistanceError,
                                               expDistanceLoss,
                                               expDistanceError,
                                               negExpDistanceLoss,
                                               negExpDistanceError)

from generic_pose.losses.quaternion_loss import quaternionLoss, quaternionError
from generic_pose.utils.pose_processing import quatAngularDiffDot, quatAngularDiffBatch

def evaluateRegression(model, 
                       query_imgs, query_quats, 
                       optimizer=None, 
                       retain_graph = False, 
                       calc_metrics=False):
    model.eval()
    if(optimizer is not None):
        optimizer.zero_grad()
    
    rendered_imgs = query_imgs[:,3:]
    query_imgs = to_var(query_imgs[:,:3])
    query_quats = to_var(query_quats).detach()
  
    if(rendered_imgs.nelement()):
        query_imgs = torch.cat((to_var(real_imgs), query_imgs))
        query_quats = torch.cat((to_var(torch.zeros([rendered_imgs.shape[0], 4], dtype=torch.float64)), 
                                query_quats))
    
    query_features = model.queryFeatures(query_imgs)
    quat_est = model.compare_network(query_features)
    loss = quaternionLoss(quat_est, query_quats)

    if(optimizer is not None):
        model.train()
        loss.backward(retain_graph=retain_graph)
        #if(clip is not None):
        #    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

    results = {}
    results['loss'] = float(to_np(loss))

    if(calc_metrics):
        ang_errs = quaternionError(quat_est, query_quats)
        results['errs_vec'] = ang_errs*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))

    del query_quats, query_imgs, query_features
    del quat_est, loss
    torch.cuda.empty_cache()

    return results



