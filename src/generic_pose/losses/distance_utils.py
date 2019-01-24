# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn 
"""
import time
import numpy as np
import torch
from functools import partial
from tqdm import tqdm

from generic_pose.utils import to_var, to_np
from generic_pose.losses.distance_loss import (rawDistanceLoss,
                                               rawDistanceError,
                                               logDistanceLoss,
                                               logDistanceError,
                                               expDistanceLoss,
                                               expDistanceError,
                                               negExpDistanceLoss,
                                               negExpDistanceError)

from generic_pose.utils.pose_processing import tensorAngularDiff, tensorAngularAllDiffs, getGaussianKernal
from generic_pose.utils.pose_processing import quatAngularDiffDot
from generic_pose.eval.posecnn_eval import evaluateQuat, getYCBThresholds
from quat_math import invAngularPDF, quaternion_from_matrix, quatAngularDiff

def getDistanceLoss(loss_type, falloff_angle):
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
    
    return distanceLoss, distanceError, dist_sign

def getFeatures(model, grid_imgs, image_chunk_size = 500):
    grid_img_chunks = torch.split(grid_imgs, image_chunk_size)
    grid_features = []
    for imgs in grid_img_chunks:
        grid_features.append(model.originFeatures(imgs).detach())
    grid_features = torch.cat(grid_features)
    return grid_features

def evaluateDataset(model, data_loader,
                    grid_quats, grid_imgs,
                    loss_type = 'exp',
                    falloff_angle = 20*np.pi/180,
                    image_chunk_size = 500,
                    num_indices = float('inf'),
                    sigma = None, 
                    save_output = False,
                    ):
    with torch.no_grad():
        if(sigma is not None):
            kernal = getGaussianKernal(grid_quats, sigma)
        else:
            kernal = None

        dataset = data_loader.dataset
        points = dataset.getObjectPoints()
        cls = dataset.getObjectName()
        use_sym = cls == '024_bowl' or cls == '036_wood_block' or cls == '061_foam_brick'
        thresholds = getYCBThresholds()
        thresh = thresholds[dataset.obj]
        
        model.eval()
        grid_imgs = to_var(grid_imgs)
        grid_features = getFeatures(model, grid_imgs, image_chunk_size)

        agg_metrics = {} 
        agg_metrics['top_idx'] = [] 
        agg_metrics['true_idx'] = [] 
        agg_metrics['rank_gt'] = [] 
        agg_metrics['rank_top'] = [] 
        agg_metrics['output_gt'] = [] 
        agg_metrics['dist_top'] = [] 
        agg_metrics['error_add'] = [] 
        agg_metrics['error_add_pcnn'] = [] 
        agg_metrics['dist_pcnn'] = [] 
        agg_metrics['idx'] = []
        
        if(save_output):
            agg_metrics['outputs'] = [] 
        
        dataset = data_loader.dataset 

        pbar = tqdm(enumerate(data_loader), total = min(len(data_loader), num_indices))
        #t = time.time()

        for batch_idx, (query_imgs, query_quats, _1, indices) in pbar:
            del _1 
            #print('Load Time: ', time.time()-t)
            #t = time.time()
            query_imgs = to_var(query_imgs)
            dist_true = np.split(quatAngularDiffDot(to_np(query_quats), grid_quats), query_quats.shape[0])
            #print('Convert Time: ', time.time()-t)
            #t = time.time()
            for q_img, d_true, q_true, idx in zip(query_imgs, dist_true, query_quats, indices):
                if(torch.norm(q_true) > 0):
                    _, metrics = evaluateImage(model, loss_type, falloff_angle, grid_features, 
                                                    q_img.unsqueeze(0), d_true.flatten(), 
                                                    kernal = kernal,
                                                    save_output = save_output)
                    for k,v in agg_metrics.items():
                        if(k in metrics.keys()):
                            v.append(metrics[k])
                       
                    q_pred = grid_quats[metrics['top_idx']]
                    mat_pcnn = dataset.getTrans(idx, use_gt=False)
                    mat_true = dataset.getTrans(idx, use_gt=True)
                    if(mat_pcnn is not None):
                        err = evaluateQuat(q_true, q_pred, points, use_sym=use_sym, 
                                t_pred = mat_pcnn[:3,3], t_true = mat_true[:3, 3])

                        q_pcnn = quaternion_from_matrix(mat_pcnn)
                        err_pcnn = evaluateQuat(q_true, q_pcnn, points, use_sym=use_sym, 
                                t_pred = mat_pcnn[:3,3], t_true = mat_true[:3, 3]) 
                        dist_pcnn = quatAngularDiff(q_pcnn, q_true)*180/np.pi
                    else:
                        err = float('NAN')
                        err_pcnn = float('NAN')
                        dist_pcnn = float('NAN')
                    agg_metrics['error_add'].append(err)
                    agg_metrics['error_add_pcnn'].append(err_pcnn)
                    agg_metrics['dist_pcnn'].append(dist_pcnn)
                    agg_metrics['idx'].append(idx)
                else:
                    for k,v in agg_metrics.items():
                        v.append(float('NAN'))


            #print('Eval Time: ', time.time()-t)
            #t = time.time()
            if(batch_idx >= num_indices):
                break
    return agg_metrics 

def evaluateImage(model, loss_type, falloff_angle,
                  grid_features, 
                  query_img, dist_true,
                  calc_metrics = True,
                  calc_loss = False, 
                  kernal = None,
                  save_output = False,
                  ):
    loss_function, error_function, dist_sign = getDistanceLoss(loss_type, falloff_angle)
    #t = time.time()
    query_features = model.queryFeatures(query_img).repeat(grid_features.shape[0],1)
    #print('Feature Time: ', time.time()-t)
    #t = time.time()
    dist_est = model.compare_network(grid_features,query_features)
    if(kernal is not None):
        dist_est = torch.mm(test, kernal_norm)
    dist_est = dist_est.flatten()
    #print('Forward Time: ', time.time()-t)
    #t = time.time()
    if(calc_loss):
        loss = loss_function(dist_est, to_var(torch.tensor(dist_true)), reduction='none')
    else:
        loss = None
    metrics = {}

    #print('Dist Time: ', time.time()-t)
    #t = time.time()
    if(calc_metrics):
        #dist_est = to_np(dist_est.detach())
        top_idx = torch.argmin(dist_sign*dist_est)
        top_idx = int(top_idx)
        metrics['top_idx'] =top_idx
        #print('top_idx Time: ', time.time()-t)
        #t = time.time()
        true_idx = int(np.argmin(dist_true))
        #print(dist_est.shape, dist_true.shape, top_idx, true_idx)
        metrics['true_idx'] = true_idx
        #print('true_idx Time: ', time.time()-t)
        #t = time.time()
        metrics['rank_gt'] = int(to_np((torch.sort(dist_sign*dist_est)[1] == true_idx).nonzero()[0][0]))
        #print('rank_gt Time: ', time.time()-t)
        #t = time.time()
        metrics['rank_top'] = int(np.nonzero(np.argsort(dist_true) == top_idx)[0][0])
        #print('rank_top Time: ', time.time()-t)
        #t = time.time()
        metrics['output_gt'] = float(to_np(dist_est[true_idx]))
        #print('output_gt Time: ', time.time()-t)
        #t = time.time()
        metrics['dist_top'] = dist_true[top_idx]*180/np.pi
        #print('dist Time: ', time.time()-t)
        #t = time.time()
        if(save_output):
            metrics['outputs'] = to_np(dist_est)
    #print('Metric Time: ', time.time()-t)
    #t = time.time()

    return loss, metrics  

def sampleIndices(num_samples, dists, top_n = 0, sampling_distribution = None):
    if(sampling_distribution is not None):
        p = invAngularPDF(dists, 1.0)
        p *= sampling_distribution(dists)
        p /= np.sum(p)
    else:
        p = None
            
    if(top_n > 0): 
        sorted_indices = np.argsort(dists)[::-1]
        top_indices = sorted_indices[:top_n]
        if(type(p) is np.ndarray):
            p = p(sorted_indices[top_n:])
    else:
        sorted_indices = np.arange(dists.size, dtype=int)
        top_indices = []

    sampled_indices = np.random.choice(sorted_indices[top_n:], num_samples - top_n, replace=False, p=p)    
    pool_indices = np.concatenate([top_indices, sampled_indices]).astype(int)
    return pool_indices

def pool2BatchIndices(pool_indices, grid_size):
    pool_indices = np.concatenate([top_indices, sampled_indices])
    grid_indices = np.remainder(pool_indices, grid_size)
    query_indices = (pool_indices / grid_size).astype(int)
    return grid_indices, query_indices

def getIndices(model, 
               query_imgs, query_quats,
               grid_imgs, grid_quats,
               loss_type='exp',
               falloff_angle = 20*np.pi/180 ,
               num_indices = 256,
               image_chunk_size = 500,
               per_instance = True,
               sample_by_loss = False,
               top_n = 0,
               calc_metrics = False,
               sampling_distribution = None):
   
    dist_true = quatAngularDiffDot(query_quats, grid_quats)
    batch_size = query_imgs.shape[0]

    grid_indices = []
    query_indices = []
    agg_metrics = {} 

    if(calc_metrics or sample_by_loss):
        with torch.no_grad():
            model.eval()
            grid_imgs = to_var(grid_imgs)
            grid_features = getFeatures(model, grid_imgs, image_chunk_size = image_chunk_size)

    if(calc_metrics):
        agg_metrics['top_idx'] = [] 
        agg_metrics['true_idx'] = [] 
        agg_metrics['rank_gt'] = [] 
        agg_metrics['rank_top'] = [] 
        agg_metrics['output_gt'] = [] 
        agg_metrics['dist_top'] = [] 
    for b_idx, (q_img, d_true) in enumerate(zip(query_imgs, dist_true)):
        num_samples = num_indices//batch_size
        if(calc_metrics or sample_by_loss):
            loss, metrics = evaluateImage(model, loss_type, falloff_angle, grid_features, 
                                          q_img.unsqueeze(0), d_true)
            for k,v in agg_metrics.items(): 
                v.append(metrics[k]) 
            
        if(sample_by_loss):
            dists = -loss
        else:
            dists = d_true
        grid_indices.extend(sampleIndices(num_samples, dists, top_n, sampling_distribution))
        query_indices.extend([b_idx for _ in range(num_samples)])

    return grid_indices, query_indices, agg_metrics

