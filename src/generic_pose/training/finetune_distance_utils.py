# -*- coding: utf-8 -*-
"""
@author: bokorn
"""
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

def evaluateRenderedDistance(model, grid, renderer,
                             query_imgs, query_quats, 
                             grid_imgs, grid_quats,
                             loss_type='exp',
                             optimizer=None, 
                             retain_graph = False, 
                             disp_metrics=False, 
                             falloff_angle = np.pi/4,
                             image_chunk_size = 500,
                             feature_chunk_size = 5000,
                             num_indices = 256,
                             per_instance = False,
                             sample_by_loss = False,
                             top_n = 0,
                             uniform_prop = .5,
                             loss_temperature = None,
                             sampling_distribution = None):
    if(loss_type.lower() == 'exp'):
        distanceLoss = partial(expDistanceLoss, falloff_angle=falloff_angle)
        distanceError = partial(expDistanceError, falloff_angle=falloff_angle)
        dist_sign = -1.0
    elif(loss_type.lower() == 'log'):
        distanceLoss = partial(logDistanceLoss, falloff_angle=falloff_angle)
        distanceError = partial(logDistanceError, falloff_angle=falloff_angle)
        dist_sign = 1.0
    elif(loss_type.lower() == 'negexp'):
        distanceLoss = partial(negExpDistanceLoss, falloff_angle=falloff_angle)
        distanceError = partial(negExpDistanceError, falloff_angle=falloff_angle)
        dist_sign = 1.0
    elif(loss_type.lower() == 'raw'):
        distanceLoss = partial(rawDistanceLoss, falloff_angle=falloff_angle)
        distanceError = partial(rawDistanceError, falloff_angle=falloff_angle)
        dist_sign = 1.0
    else:
        raise ValueError('Invalid Loss Type: {}'.format(loss_type.lower()))
    model.eval()

    if(optimizer is not None):
        optimizer.zero_grad()
    #print('pre_util')
    #import IPython; IPython.embed()
    real_imgs = query_imgs[:,:3]
    rendered_imgs = query_imgs[:,3:]
    grid_indices, query_indices, results = hardExampleMining(model,
                                                             real_imgs, query_quats,
                                                             grid_imgs, grid_quats,
                                                             loss_function=distanceLoss,
                                                             image_chunk_size = image_chunk_size,
                                                             feature_chunk_size = feature_chunk_size,
                                                             num_indices = num_indices,
                                                             per_instance = per_instance,
                                                             sample_by_loss = sample_by_loss,
                                                             top_n = top_n,
                                                             uniform_prop = uniform_prop,
                                                             loss_temperature = loss_temperature,
                                                             disp_metrics = disp_metrics,
                                                             dist_sign = dist_sign,
                                                             sampling_distribution = sampling_distribution)
    torch.cuda.empty_cache()
    #import IPython; IPython.embed()

    dist_true = to_var(torch.tensor(quatAngularDiffBatch(to_np(query_quats[query_indices]), 
                                                         grid_quats[grid_indices]))).detach()
#    quat_true = to_var(torch.tensor(quatDiffBatch(to_np(query_quats[query_indices]), 
#                                                       grid_quats[grid_indices]))).detach()

    grid_img_samples = to_var(grid_imgs[grid_indices])
    query_img_samples = to_var(real_imgs[query_indices])

    if(len(rendered_imgs)):
        grid_img_samples = torch.cat((to_var(rendered_imgs), grid_img_samples))
        query_img_samples = torch.cat((to_var(real_imgs), query_img_samples))
        #zero_quats = to_var(torch.zeros((real_imgs.shape[0],4), dtype=torch.float64)).detach()
        #zero_quats[:,3] = 1
        #quat_true = torch.cat((zero_quats, quat_true))
        dist_true = torch.cat((to_var(torch.zeros(real_imgs.shape[0], dtype=torch.float64)).detach(), dist_true))
    grid_features = model.originFeatures(grid_img_samples)
    query_features = model.queryFeatures(query_img_samples)
    dist_est = model.compare_network(grid_features, query_features)
    #loss = distanceLoss(dist_est.flatten(), quat_true, reduction='elementwise_mean')
    loss = distanceLoss(dist_est.flatten(), dist_true, reduction='elementwise_mean')

    if(optimizer is not None):
        model.train()
        loss.backward(retain_graph=retain_graph)
        #if(clip is not None):
        #    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

    results['loss'] = float(to_np(loss))

    if(disp_metrics):
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

    del grid_imgs, grid_img_samples, grid_features
    del query_imgs, query_img_samples, query_features
    #del loss, dist_est, quat_true
    del loss, dist_est, dist_true
    torch.cuda.empty_cache()

    return results


def hardExampleMining(model,
                      query_imgs, query_quats,
                      grid_imgs, grid_quats,
                      loss_function=expDistanceLoss,
                      image_chunk_size = 500,
                      feature_chunk_size = 5000,
                      num_indices = 32,
                      per_instance = False,
                      sample_by_loss = False,
                      top_n = 0,
                      uniform_prop = .5,
                      loss_temperature = None,
                      disp_metrics = False,
                      dist_sign = 1.0,
                      sampling_distribution = None):
    model.eval()

    grid_imgs = to_var(grid_imgs)
    query_imgs = to_var(query_imgs)
    query_quats = to_np(query_quats)

    grid_size = grid_quats.shape[0]
    batch_size = query_quats.shape[0]
    pool_size = grid_size*batch_size
    
    metrics = {}

    if(disp_metrics or sample_by_loss):
        with torch.no_grad():
            grid_img_chunks = torch.split(grid_imgs, image_chunk_size)
            grid_features = []
            for imgs in grid_img_chunks:
                grid_features.append(model.originFeatures(imgs).detach())
                torch.cuda.empty_cache()
            grid_features = torch.cat(grid_features)

            # Duplicating query data [1, 2, 3, ... 1, 2, 3, ...]
            grid_features = grid_features.repeat(batch_size,1)
            grid_quats_rep = np.tile(grid_quats, (batch_size,1))

            query_features = model.queryFeatures(query_imgs).detach()
            torch.cuda.empty_cache()
            
            # Duplicating query data [1, 1, 1, ... 2, 2, 2, ...]
            query_features = query_features.repeat(1,grid_size).view(pool_size,-1)
            query_quats_rep = np.repeat(query_quats, grid_size, axis = 0)

            grid_feature_chunks = torch.split(grid_features, feature_chunk_size)
            query_feature_chunks = torch.split(query_features, feature_chunk_size)
            dist_est = []
            for gf, qf in zip(grid_feature_chunks, query_feature_chunks):
                dist_est.append(model.compare_network(gf,qf).detach())
                torch.cuda.empty_cache()
            dist_est = torch.cat(dist_est).flatten()
            #quat_true = to_var(torch.tensor(quatDiffBatch(query_quats_rep, grid_quats_rep)))
            #loss = loss_function(dist_est, quat_true, reduction='none')
            dist_true = to_var(torch.tensor(quatAngularDiffDot(query_quats, grid_quats)))
            loss = loss_function(dist_est, dist_true, reduction='none')
            
            if(disp_metrics):
                rank_gt = []
                rank_top = []
                output_gt = []
                dist_top = []
                dist_est_chunks = torch.split(dist_est, grid_size)
                #quat_true_chunks = torch.split(quat_true, grid_size)
                dist_true_chunks = torch.split(dist_true, grid_size)
                for d_est, d_true in zip(dist_est_chunks, dist_true_chunks):
                    d_est = to_np(d_est.detach())
                    #true_angles = quaternionAngles(q_true)   
                    true_angles = to_np(d_true)   
                    top_idx = np.argmin(dist_sign*d_est)
                    true_idx = np.argmin(true_angles)
                    rank_gt.append(np.nonzero(np.argsort(dist_sign*d_est) == true_idx)[0][0])
                    rank_top.append(np.nonzero(np.argsort(true_angles) == top_idx)[0][0])
                    output_gt.append(d_est[true_idx])
                    dist_top.append(true_angles[top_idx]*180/np.pi)
                metrics['rank_gt'] = np.mean(rank_gt)
                metrics['rank_top'] = np.mean(rank_top)
                metrics['output_gt'] = np.mean(output_gt)
                metrics['dist_top'] = np.mean(dist_top)
                
            del grid_features, grid_feature_chunks
            del query_features, query_feature_chunks 
            del gf, qf, dist_est, dist_true, 
            #del gf, qf, dist_est, quat_true, 
            #torch.cuda.empty_cache()
            
            #quat_true = quat_true.cpu()
            grid_imgs = grid_imgs.cpu()
            query_imgs = query_imgs.cpu()
            dist = to_np(loss)
            del loss
    elif(loss_temperature is not None or uniform_prop < 1.0 or top_n > 0):
        dist = quatAngularDiffDot(query_quats, grid_quats) 

    if(sampling_distribution is not None):
        dist_true = quatAngularDiffDot(query_quats, grid_quats)
        p = invAngularPDF(dist_true, 1.0)
        p *= sampling_distribution(dist_true)
        p /= np.sum(p)
    else:
        p = None

    if(loss_temperature is not None): 
        #p = np.exp(loss_temperature*to_np(loss))
        p = np.exp(loss_temperature*dist)
        p /= p.sum()
        assert sum(p>0) > num_indices, 'Temperature parameter to high, insufficnent non-zero probabilities'
        #pool_indices = np.random.choice(np.arange(loss.shape[0]), num_indices, replace = False, p=p)
        pool_indices = np.random.choice(np.arange(dist.shape[0]), num_indices, replace = False, p=p)
        #del loss
    elif(uniform_prop < 1.0 or top_n > 0):
        if(top_n > 0):
            if(per_instance):
                cutoff_idx = top_n * batch_size
            else:
                cutoff_idx = top_n
        else:
            cutoff_idx = int(np.floor(num_indices*(1.0-uniform_prop)))
        
        num_uniform = num_indices - cutoff_idx
        #loss_indices = to_np(torch.sort(loss, descending=True)[1])

        if(per_instance):
            indices = np.arange(dist.shape[0])
            top_indices = []
            uniform_indices = []
            if(type(p) is np.ndarray):
                p = np.split(p, batch_size)
            else:
                p = itertools.repeat(p)
            c_rem = cutoff_idx
            n_rem = num_uniform
            for batch_idx, (batch_dist, batch_p) in enumerate(zip(np.split(dist, batch_size), p)):
                b_rem = batch_size-batch_idx
                c_idx = int(np.ceil(c_rem/b_rem))
                n_uni = int(np.ceil(n_rem/b_rem))
                c_rem -= c_idx
                n_rem -= n_uni
                loss_indices = np.argsort(batch_dist)[::-1]
                top_indices.extend(loss_indices[:c_idx] + batch_idx*batch_size)
                if(type(p) is np.ndarray):
                    batch_p = batch_p(loss_indices[c_idx:])
                uniform_indices.extend(np.random.choice(loss_indices[cutoff_idx:], 
                    n_uni, replace=False, p=batch_p) + batch_idx*batch_size)
        else:
            loss_indices = np.argsort(dist)[::-1]
            top_indices = loss_indices[:cutoff_idx]
            if(type(p) is np.ndarray):
                p = p(loss_indices[cutoff_idx:])
            uniform_indices = np.random.choice(loss_indices[cutoff_idx:], num_uniform, replace=False, p=p)
        pool_indices = np.concatenate([top_indices, uniform_indices])
        #del dist
    else:
        if(per_instance):
            pool_indices = []
            if(type(p) is np.ndarray):
                p = np.split(p, batch_size)
            else:
                p = itertools.repeat(p)
            n_rem = num_indices
            for batch_idx, (batch_dist, batch_p) in enumerate(zip(np.split(dist, batch_size), p)):
                b_rem = batch_size-batch_idx
                n = int(np.ceil(n_rem/b_rem))
                n_rem -= n
                pool_indices.extend(np.random.choice(batch_size, n, replace=False, p=batch_p) \
                        + batch_idx*batch_size)
 
            pool_indices = np.array(pool_indices)
        else:
            pool_indices = np.random.choice(batch_size*grid_size, num_indices, replace=False, p = p)
        
    grid_indices = np.remainder(pool_indices, grid_size)
    query_indices = (pool_indices / grid_size).astype(int)
    return grid_indices, query_indices, metrics

