# -*- coding: utf-8 -*-
"""
@author: bokorn
"""
import numpy as np

from generic_pose.utils import to_var, to_np
from generic_pose.losses.distance_loss import (rawDistanceLoss,
                                               rawDistanceError,
                                               logDistanceLoss, 
                                               logDistanceError,
                                               expDistanceLoss,
                                               expDistanceError,
                                               negExpDistanceLoss,
                                               negExpDistanceError)

from generic_pose.losses.quaternion_loss import quaternionAngles

def evaluateDistance(model, origin, query, quat_true,
                     loss_type='exp',
                     optimizer=None, 
                     retain_graph = False, 
                     disp_metrics=False, 
                     falloff_angle = np.pi/4,
                     clip=None):

    if(loss_type.lower() == 'exp'):
        distanceLoss = expDistanceLoss
        distanceError = expDistanceError
    elif(loss_type.lower() == 'log'):
        distanceLoss = logDistanceLoss
        distanceError = logDistanceError
    elif(loss_type.lower() == 'negexp'):
        distanceLoss = negExpDistanceLoss
        distanceError = negExpDistanceError
    elif(loss_type.lower() == 'raw'):
        distanceLoss = rawDistanceLoss
        distanceError = rawDistanceError
    else:
        raise ValueError('Invalid Loss Type: {}'.format(loss_type.lower()))

    origin = to_var(origin)
    query = to_var(query)
    quat_true = to_var(quat_true)

    results = {}

    if(optimizer is not None):
        optimizer.zero_grad()

    origin_features = model.features(origin)
    query_features = model.features(query)
    dist_est = model.compare_network(origin_features, 
                                     query_features)

    loss = distanceLoss(dist_est, quat_true, 
                        falloff_angle = falloff_angle)

    if(optimizer is not None):
        loss.backward(retain_graph=retain_graph)
        if(clip is not None):
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

    results['loss'] = loss.data[0]
    
    if(disp_metrics):
        ang_errs = distanceError(dist_est, quat_true, falloff_angle=falloff_angle)
        ang_diff = quaternionAngles(quat_true)
        results['dist_vec'] = to_np(dist_est)
        results['errs_vec'] = ang_errs*180.0/np.pi
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_err_thresh_{}'.format(int(falloff_angle*180/np.pi))] = \
                np.mean(ang_errs[ang_diff<falloff_angle])*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(origin_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))
    
    del origin, query, quat_true

    return results


