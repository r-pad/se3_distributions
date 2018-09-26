# -*- coding: utf-8 -*-
"""
@author: bokorn
"""
import numpy as np
from functools import partial
import torch

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
from generic_pose.utils.pose_processing import viewpointDiffBatch

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
                             uniform_prop = .5,
                             loss_temperature = None):
    if(loss_type.lower() == 'exp'):
        distanceLoss = partial(expDistanceLoss, falloff_angle=falloff_angle)
        distanceError = partial(expDistanceError, falloff_angle=falloff_angle)
    elif(loss_type.lower() == 'log'):
        distanceLoss = partial(logDistanceLoss, falloff_angle=falloff_angle)
        distanceError = partial(logDistanceError, falloff_angle=falloff_angle)
    elif(loss_type.lower() == 'negexp'):
        distanceLoss = partial(negExpDistanceLoss, falloff_angle=falloff_angle)
        distanceError = partial(negExpDistanceError, falloff_angle=falloff_angle)
    elif(loss_type.lower() == 'raw'):
        distanceLoss = partial(rawDistanceLoss, falloff_angle=falloff_angle)
        distanceError = partial(rawDistanceError, falloff_angle=falloff_angle)
    else:
        raise ValueError('Invalid Loss Type: {}'.format(loss_type.lower()))
    model.eval()

    results = {}
    if(optimizer is not None):
        optimizer.zero_grad()
    #print('pre_util')
    #import IPython; IPython.embed()
    grid_indices, query_indices = hardExampleMining(model,
                                                    query_imgs, query_quats,
                                                    grid_imgs, grid_quats,
                                                    loss_function=distanceLoss,
                                                    image_chunk_size = image_chunk_size,
                                                    feature_chunk_size = feature_chunk_size,
                                                    num_indices = num_indices,
                                                    uniform_prop = uniform_prop,
                                                    loss_temperature = loss_temperature)
    torch.cuda.empty_cache()
    #import IPython; IPython.embed()
        
    quat_true = to_var(torch.tensor(viewpointDiffBatch(to_np(query_quats[query_indices]), 
                                                       grid_quats[grid_indices]))).detach()

    grid_img_samples = to_var(grid_imgs[grid_indices])
    query_img_samples = to_var(query_imgs[query_indices])
    grid_features = model.originFeatures(grid_img_samples)
    query_features = model.queryFeatures(query_img_samples)
    dist_est = model.compare_network(grid_features, query_features)
    loss = distanceLoss(dist_est.flatten(), quat_true, reduce=True)

    if(optimizer is not None):
        model.train()
        loss.backward(retain_graph=retain_graph)
        #if(clip is not None):
        #    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

    results['loss'] = float(to_np(loss))

    if(disp_metrics):
        ang_errs = distanceError(dist_est, quat_true)
        ang_diff = quaternionAngles(quat_true)
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
    del loss, dist_est, quat_true
    torch.cuda.empty_cache()

    return results


def hardExampleMining(model,
                      query_imgs, query_quats,
                      grid_imgs, grid_quats,
                      loss_function=expDistanceLoss,
                      image_chunk_size = 500,
                      feature_chunk_size = 5000,
                      num_indices = 32,
                      uniform_prop = .5,
                      loss_temperature = None):
    model.eval()

    grid_imgs = to_var(grid_imgs)
    query_imgs = to_var(query_imgs)
    query_quats = to_np(query_quats)

    grid_size = grid_quats.shape[0]
    batch_size = query_quats.shape[0]
    pool_size = grid_size*batch_size


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
        quat_true = to_var(torch.tensor(viewpointDiffBatch(query_quats_rep, grid_quats_rep)))
        loss = loss_function(dist_est, quat_true, reduce=False)
        del grid_features, grid_feature_chunks
        del query_features, query_feature_chunks 
        del gf, qf, dist_est, quat_true, 
        #torch.cuda.empty_cache()
        
        #quat_true = quat_true.cpu()
        grid_imgs = grid_imgs.cpu()
        query_imgs = query_imgs.cpu()

    if(loss_temperature is None): 
        cutoff_idx = int(np.floor(num_indices*uniform_prop))
        num_uniform = int(np.ceil(num_indices*(1.0-uniform_prop)))
        loss_indices = to_np(torch.sort(loss, descending=True)[1])
        top_indices = loss_indices[:cutoff_idx]
        uniform_indices = np.random.choice(loss_indices[cutoff_idx:], num_uniform, replace=False)
        pool_indices = np.concatenate([top_indices, uniform_indices])
    else:
        p = np.exp(loss_temperature*loss)
        p /= p.sum()
        assert sum(p>0) > num_indices, 'Temperature parameter to high, insufficnent non-zero probabilities'
        pool_indices = np.random.choice(np.arange(loss.shape[0]), num_indices, replace = False, p=p)
    
    del loss
    grid_indices = np.remainder(pool_indices, grid_size)
    query_indices = (pool_indices / grid_size).astype(int)
    return grid_indices, query_indices

