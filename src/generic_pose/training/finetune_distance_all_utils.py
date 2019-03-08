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

#from generic_pose.losses.quaternion_loss import quaternionAngles
#from generic_pose.utils.pose_processing import quatDiffBatch, quatAngularDiffBatch
from generic_pose.utils.pose_processing import quatAngularDiffDot, quatAngularDiffBatch
from generic_pose.losses.distance_utils import getDistanceLoss

def evaluateRenderedDistance(model, 
                             query_imgs, grid_imgs, grid_dist,
                             loss_type='exp',
                             optimizer=None, 
                             retain_graph = False, 
                             calc_metrics=False, 
                             falloff_angle = np.pi/4,
                             ):
    distanceLoss, distanceError, dist_sign = getDistanceLoss(loss_type, falloff_angle)
    model.eval()
    if(optimizer is not None):
        optimizer.zero_grad()
           
    torch.cuda.empty_cache()
    n_query = query_imgs.shape[0]
    n_grid = grid_dist.shape[1]
    query_idxs = np.repeat(np.arange(n_query), n_grid)
    
    query_imgs = to_var(query_imgs)[query_idxs]
    grid_dist = to_var(grid_dist).view(-1)
    grid_imgs = to_var(grid_imgs).view(-1, *grid_imgs.shape[2:])
    
    query_features = model.queryFeatures(query_imgs)
    grid_features = model.originFeatures(grid_imgs)
    
    dist_est = model.compare_network(grid_features, query_features)
    loss = distanceLoss(dist_est.flatten(), grid_dist, reduction='mean')

    if(optimizer is not None):
        model.train()
        loss.backward()
        optimizer.step()
    
    results = {}
    results['loss'] = float(to_np(loss))

    if(calc_metrics):
        ang_errs = distanceError(dist_est, grid_dist)
        ang_diff = to_np(grid_dist)
        results['dist_vec'] = to_np(dist_est)
        results['errs_vec'] = ang_errs*180.0/np.pi
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_err_thresh_{}'.format(int(falloff_angle*180/np.pi))] = \
                np.mean(ang_errs[ang_diff<falloff_angle])*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(grid_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))

    del grid_imgs, grid_features, grid_dist
    del query_imgs, query_features
    del loss, dist_est
    torch.cuda.empty_cache()

    return results



