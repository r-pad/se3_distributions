# -*- coding: utf-8 -*-
"""
Created on Thursday at some later point in time
@author: bokorn 
"""
import numpy as np
import torch
from object_pose_utils.utils.pose_processing import tensorAngularDiff
from object_pose_utils.utils import to_np, to_var
import torch.nn.functional as F
from se3_distributions.losses.distance_loss import expDistanceLoss, expDistanceError


def evaluateLoss(model,  
                 features, quats, 
                 grid_vertices,
                 falloff_angle = np.pi/9.,
                 weight_top = 1.0,
                 optimizer = None,
                 grid_size = 3885,
                 calc_metrics = True,
                 retain_graph = False,
                 ):

    num_features = features.shape[0]
    rep_indices = np.repeat(np.arange(num_features), grid_size)
    dist_est = model(features)
    dist_est = dist_est.flatten()

    #dist_true = tensorAngularDiff(grid_vertices, quats[rep_indices]).view(-1, grid_size)
    #labels = torch.exp(-dist_true/falloff_angle)
    #loss = -(labels * F.log_softmax(dist_est, dim=1))

    #if(weight_top != 1.0):
    #    min_idx = torch.argmin(dist_true, dim=1)
    #    loss[min_idx] *= weight_top

    #loss = loss.mean(1).sum()

    dist_true = tensorAngularDiff(grid_vertices, quats[rep_indices])

    if(weight_top != 1.0):
        loss = expDistanceLoss(dist_est, to_var(dist_true), reduction='none')
        min_idx = torch.argmin(dist_true.view(-1, grid_size), dim=1) + torch.arange(num_features).cuda()*grid_size
        #top_idx = torch.argmax(dist_est.view(-1, grid_size), dim=1) + torch.arange(num_features).cuda()*grid_size
        loss[min_idx] *= weight_top
        #loss[top_idx] *= weight_top
        loss = loss.mean()
    else:
        loss = expDistanceLoss(dist_est, to_var(dist_true), reduction='mean')

    if(optimizer is not None):
        model.train()
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

    metrics = {}
    metrics['loss'] = float(to_np(loss))

    if(calc_metrics):
        dist_est = dist_est.view(-1, grid_size)
        dist_true = dist_true.view(-1, grid_size)

        top_idx = torch.argmax(dist_est, dim=1).detach()
        metrics['top_idx_vec'] = to_np(top_idx)
        true_idx = torch.argmin(dist_true, dim=1).detach()
        metrics['true_idx_vec'] = to_np(true_idx)

        metrics['rank_gt'] = to_np((torch.sort(dist_est, descending=True, dim=1)[1] \
                == true_idx.unsqueeze(1)).nonzero()[:,1]).mean()

        metrics['rank_top'] = to_np((torch.sort(dist_true, dim=1)[1] \
                == top_idx.unsqueeze(1)).nonzero()[:,1]).mean()
       
        dist_shape = dist_est.shape
        true_idx_flat = np.ravel_multi_index(np.stack([np.arange(dist_shape[0]), to_np(true_idx)]), dist_shape)
        top_idx_flat = np.ravel_multi_index(np.stack([np.arange(dist_shape[0]), to_np(top_idx)]), dist_shape)

        metrics['output_gt'] = to_np(dist_est.view(-1)[true_idx_flat]).mean()
        metrics['dist_gt'] = to_np(dist_true.view(-1)[true_idx_flat]).mean()*180./np.pi
        metrics['exp_output_gt'] = np.exp(-metrics['dist_gt']*np.pi/180.0/(falloff_angle)) 
        metrics['dist_top'] = to_np(dist_true.view(-1)[top_idx_flat]).mean()*180./np.pi

    return metrics  

def multiObjectLoss(model, objs, 
                 features, quats, 
                 grid_vertices,
                 falloff_angle = np.pi/9.,
                 weight_top = 1.0,
                 optimizer = None,
                 grid_size = 3885,
                 calc_metrics = True,
                 retain_graph = False,
                 ):

    num_features = features.shape[0]
    rep_indices = np.repeat(np.arange(num_features), grid_size)
    dist_est = model(features).view(num_features, -1, grid_size)
    dist_est = dist_est[:,objs]
    #dist_est = torch.gather(dist_est, 1, objs[rep_indices])
    dist_est = dist_est.flatten()

    dist_true = tensorAngularDiff(grid_vertices, quats[rep_indices])
    import IPython; IPython.embed()

    if(weight_top != 1.0):
        loss = expDistanceLoss(dist_est, to_var(dist_true), reduction='none')
        min_idx = torch.argmin(dist_true.view(-1, grid_size), dim=1) + torch.arange(num_features).cuda()*grid_size
        loss[min_idx] *= weight_top
        loss = loss.mean()
    else:
        loss = expDistanceLoss(dist_est, to_var(dist_true), reduction='mean')

    if(optimizer is not None):
        model.train()
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

    metrics = {}
    metrics['loss'] = float(to_np(loss))
    
    if(calc_metrics):
        for obj_id in torch.unique(objs):
            obj_idxs = (objs == obj_id).nonzero()[:,0]
            dist_est_obj = dist_est.view(-1, grid_size)[obj_idxs]
            dist_true_obj = dist_true.view(-1, grid_size)[obj_idxs]
            top_idx = torch.argmax(dist_est_obj, dim=1).detach()
            metrics['{}_top_idx_vec'.format(obj_id)] = to_np(top_idx)
            true_idx = torch.argmin(dist_true_obj, dim=1).detach()
            metrics['{}_true_idx_vec'.format(obj_id)] = to_np(true_idx)

            metrics['{}_rank_gt'.format(obj_id)] = to_np((torch.sort(dist_est_obj, descending=True, dim=1)[1] \
                    == true_idx.unsqueeze(1)).nonzero()[:,1]).mean()

            metrics['{}_rank_top'.format(obj_id)] = to_np((torch.sort(dist_true_obj, dim=1)[1] \
                    == top_idx.unsqueeze(1)).nonzero()[:,1]).mean()
             
            dist_shape = dist_est_obj.shape
            true_idx_flat = np.ravel_multi_index(np.stack([np.arange(dist_shape[0]), to_np(true_idx)]), dist_shape)
            top_idx_flat = np.ravel_multi_index(np.stack([np.arange(dist_shape[0]), to_np(top_idx)]), dist_shape)

            metrics['{}_output_gt'.format(obj_id)] = to_np(dist_est_obj.view(-1)[true_idx_flat]).mean()
            metrics['{}_dist_gt'.format(obj_id)] = to_np(dist_true_obj.view(-1)[true_idx_flat]).mean()*180./np.pi
            metrics['{}_dist_top'.format(obj_id)] = to_np(dist_true_obj.view(-1)[top_idx_flat]).mean()*180./np.pi

    return metrics  


