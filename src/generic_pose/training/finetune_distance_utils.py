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
from generic_pose.utils.pose_processing import quatAngularDiffDot, quatAngularDiffBatch, symmetricAngularDistance
from generic_pose.losses.distance_utils import getDistanceLoss, getIndices

def evaluateRenderedDistance(model, 
                             query_imgs, query_quats, 
                             grid_imgs, grid_quats,
                             loss_type='exp',
                             optimizer=None, 
                             retain_graph = False, 
                             calc_metrics=False, 
                             falloff_angle = np.pi/4,
                             num_indices = 256,
                             image_chunk_size = 500,
                             per_instance = False,
                             sample_by_loss = False,
                             top_n = 0,
                             sampling_distribution = None,
                             axes_of_sym = [],
                             angles_of_sym = [],
                             ):
    #t = time.time()
    distanceLoss, distanceError, dist_sign = getDistanceLoss(loss_type, falloff_angle)
    model.eval()
    #print("Get Loss:", time.time()-t)
    #t = time.time()
    if(optimizer is not None):
        optimizer.zero_grad()
    #print('pre_util')
    #import IPython; IPython.embed()
    #print("Reset Optimizer:", time.time()-t)
    #t = time.time()
    if(query_imgs.shape[1] in [3, 6]):
        split_idx = 3
    elif(query_imgs.shape[1] in [4, 8]):
        split_idx = 4
    else:
        raise ValueError('Invalid Number of Image Channels: {}, Must be [3,4,6,8]'.format(query_imgs.shape[1])) 
    real_imgs = to_var(query_imgs[:,:split_idx])
    rendered_imgs = query_imgs[:,split_idx:]
    query_quats = torch.tensor(query_quats).float()
    grid_quats = torch.tensor(grid_quats).float()
    #print("Get Images:", time.time()-t)
    #t = time.time()
    
    grid_indices, query_indices, results = getIndices(model,
                                                      real_imgs, query_quats,
                                                      grid_imgs, grid_quats,
                                                      loss_type = loss_type,
                                                      falloff_angle = falloff_angle,
                                                      num_indices = num_indices,
                                                      image_chunk_size = image_chunk_size,
                                                      per_instance = per_instance,
                                                      sample_by_loss = sample_by_loss,
                                                      top_n = top_n,
                                                      calc_metrics = calc_metrics,
                                                      sampling_distribution = sampling_distribution,
                                                      axes_of_sym = axes_of_sym,
                                                      angles_of_sym = angles_of_sym, 
                                                      )

    for k,v in results.items(): 
        results[k] = np.mean(v) 
            
    #print("Calculate Samples:", time.time()-t)
    #t = time.time()
    torch.cuda.empty_cache()
    #import IPython; IPython.embed()
#    dist_true = to_var(torch.tensor(quatAngularDiffBatch(query_quats[query_indices], 
#                                                         grid_quats[grid_indices]))).detach()
    dist_true = to_var(symmetricAngularDistance(query_quats[query_indices], grid_quats[grid_indices], 
                                                axes_of_sym, angles_of_sym)).detach()
#    quat_true = to_var(torch.tensor(quatDiffBatch(to_np(query_quats[query_indices]), 
#                                                       grid_quats[grid_indices]))).detach()
    #print("Calc Diff:", time.time()-t)
    #t = time.time()

    grid_img_samples = to_var(grid_imgs[grid_indices])
    query_img_samples = real_imgs[query_indices]
    
    #print("Convert Samples:", time.time()-t)
    #t = time.time()

    if(rendered_imgs.nelement()):
        grid_img_samples = torch.cat((to_var(rendered_imgs), grid_img_samples))
        query_img_samples = torch.cat((to_var(real_imgs), query_img_samples))
        #zero_quats = to_var(torch.zeros((real_imgs.shape[0],4), dtype=torch.float64)).detach()
        #zero_quats[:,3] = 1
        #quat_true = torch.cat((zero_quats, quat_true))
        dist_true = torch.cat((to_var(torch.zeros(real_imgs.shape[0], dtype=torch.float32)).detach(), dist_true))
    #print("Cat Exact:", time.time()-t)
    #t = time.time()
    grid_features = model.originFeatures(grid_img_samples)
    query_features = model.queryFeatures(query_img_samples)
    #print("Calc Features:", time.time()-t)
    #t = time.time()
    dist_est = model.compare_network(grid_features, query_features)
    #loss = distanceLoss(dist_est.flatten(), quat_true, reduction='elementwise_mean')
    #print("Est Dist:", time.time()-t)
    #t = time.time()
    loss = distanceLoss(dist_est.flatten(), dist_true, reduction='mean')
    #print("Calc Loss:", time.time()-t)
    #t = time.time()

    if(optimizer is not None):
        model.train()
        loss.backward(retain_graph=retain_graph)
        #if(clip is not None):
        #    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

    #print("Backprop:", time.time()-t)
    #t = time.time()
    results['loss'] = float(to_np(loss))

    if(calc_metrics):
        #ang_errs = distanceError(dist_est, quat_true)
        #ang_diff = quaternionAngles(quat_true)
        ang_errs = distanceError(dist_est, dist_true)
        ang_diff = to_np(dist_true)
        results['dist_vec'] = to_np(dist_est)
        results['errs_vec'] = ang_errs*180.0/np.pi
        results['diff_vec'] = ang_diff*180.0/np.pi
        results['mean_err'] = np.mean(ang_errs)*180.0/np.pi
        results['mean_err_thresh_{}'.format(int(falloff_angle*180/np.pi))] = \
                np.mean(ang_errs[ang_diff<falloff_angle])*180.0/np.pi
        results['mean_origin_features'] = np.mean(np.abs(to_np(grid_features)))
        results['mean_query_features'] = np.mean(np.abs(to_np(query_features)))
    #print("Display Metrics:", time.time()-t)
    #t = time.time()

    del grid_quats, grid_imgs, grid_img_samples, grid_features
    del query_quats, query_imgs, query_img_samples, query_features
    #del loss, dist_est, quat_true
    del loss, dist_est, dist_true
    torch.cuda.empty_cache()
    #print("Cleanup:", time.time()-t)
    #t = time.time()

    return results



