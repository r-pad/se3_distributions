# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn 
"""
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
from se3_distributions.utils import to_var, to_np
from se3_distributions.eval.posecnn_eval import evaluateQuat, getYCBThresholds
from se3_distributions.utils.pose_processing import tensorAngularDiff, tensorAngularAllDiffs, getGaussianKernal
from quat_math import invAngularPDF, quaternion_from_matrix, quatAngularDiff
from enum import Enum
class LossType(Enum):
    CrossEntropy = 1
    MSE = 2

def vertexDistanceLoss(pred_cls, label_qs, verts, falloff_angle = np.pi/9.0,
                       loss_type = LossType.CrossEntropy):
    labels = to_var(torch.exp(-tensorAngularAllDiffs(label_qs,verts)/falloff_angle)).float()
    if(loss_type == LossType.CrossEntropy):
        loss = -(labels * F.log_softmax(pred_cls, dim=1)).mean()
    elif(loss_type == LossType.MSE):
        loss = F.mse_loss(pred_cls, labels, reduction='mean')
    else:
        raise TypeError('Invalid Lose Type: {}'.format(loss_type))

    return loss

def vertexDistanceMetrics(pred_cls, label_qs, verts, add_data = None, kernal = None):
    with torch.no_grad():
        if(kernal is not None):
            pred_cls = torch.mm(pred_cls, kernal)
        
        metrics = {}
        top_idxs = torch.argmax(pred_cls, dim=1)
        top_qs = verts[top_idxs, :]
        pred_sort = torch.argsort(pred_cls, dim=1, descending=True)
        
        true_dists = tensorAngularAllDiffs(label_qs,verts)
        true_sort = torch.argsort(true_dists, dim=1, descending=False)
        true_idxs = torch.argmin(tensorAngularAllDiffs(label_qs,verts), dim=1)
        
        metrics['dist_top'] = tensorAngularDiff(label_qs, top_qs)*180/np.pi
        metrics['gt_rank'] = torch.nonzero(pred_sort == true_idxs.unsqueeze(1))[:,1]
        metrics['top_rank'] = torch.nonzero(true_sort == top_idxs.unsqueeze(1))[:,1] 
        metrics['top_idx'] = top_idxs
        
        if(add_data is not None):
            metrics['add_error'] = []
            metrics['add_accuracy'] = []
            points = add_data['points']
            use_sym = add_data['use_sym']
            for q_true, q_pred, t_true, t_pred in zip(label_qs, top_qs, 
                add_data['trans_true'], add_data['trans_pcnn']):
                if(t_pred is not None):
                    add_err = evaluateQuat(to_np(q_true), to_np(q_pred), points, use_sym=use_sym, 
                                           t_pred = t_pred, t_true = t_true)
                    metrics['add_error'].append(add_err)
                    metrics['add_accuracy'].append(add_err < add_data['threshold'])
                #else:
                    #pass
                    #metrics['add_error'].append(float('NAN'))
    return metrics

def tripletLoss(pred_cls, label_qs, verts):
    diff = tensorAngularDiff(label_qs, verts)


def evaluateDataset(model, data_loader,
                    base_vertices,
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
        #add_data = {'points':points,
        #            'use_sym':use_sym,
        #            'threshold':thresh}
        #add_data = None 
        model.eval()

        agg_metrics = {} 
        agg_metrics['top_idx'] = [] 
        agg_metrics['true_idx'] = [] 
        agg_metrics['rank_gt'] = [] 
        agg_metrics['rank_top'] = [] 
        agg_metrics['dist_top'] = [] 
        agg_metrics['error_add'] = [] 
        agg_metrics['error_add_pcnn'] = [] 
        agg_metrics['dist_pcnn'] = [] 
        agg_metrics['quat_gt'] = []
        agg_metrics['quat_pcnn'] = []
        agg_metrics['idx'] = []
        
        if(save_output):
            agg_metrics['outputs'] = [] 
        
        base_vertices = to_var(torch.tensor(base_vertices))
        dataset = data_loader.dataset 

        pbar = tqdm(enumerate(data_loader), total = len(data_loader))
        #t = time.time()

        for batch_idx, (query_imgs, query_quats, _1, indices) in pbar:
            del _1 
            #print('Load Time: ', time.time()-t)
            #t = time.time()
            query_imgs = to_var(query_imgs)
            #print('Convert Time: ', time.time()-t)
            #t = time.time()
            pred_cls = model(to_var(query_imgs))
            for outputs, q_true, idx in zip(pred_cls, query_quats, indices):
                if(torch.norm(q_true) > 0):
                    metrics = vertexDistanceMetrics(outputs.unsqueeze(0), 
                            to_var(q_true).unsqueeze(0), base_vertices, 
                            add_data = None, kernal = kernal)
                    for k,v in agg_metrics.items():
                        if(k in metrics.keys()):
                            v.append(to_np(metrics[k]))
                       
                    q_pred = to_np(base_vertices[metrics['top_idx']].flatten())
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
                        q_pcnn = np.zeros(4) 
                    agg_metrics['error_add'].append(err)
                    agg_metrics['error_add_pcnn'].append(err_pcnn)
                    agg_metrics['dist_pcnn'].append(dist_pcnn)
                    agg_metrics['quat_gt'].append(to_np(q_true))
                    agg_metrics['quat_pcnn'].append(q_pcnn)
                    agg_metrics['idx'].append(idx)
                    if(save_output):
                        agg_metrics['outputs'].append(to_np(outputs.detatch()))
                else:
                    for k,v in agg_metrics.items():
                        v.append(float('NAN'))

    return agg_metrics 

