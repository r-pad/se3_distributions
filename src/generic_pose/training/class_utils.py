# -*- coding: utf-8 -*-
"""
@author: bokorn
"""
import time
import numpy as np
from functools import partial
import torch
import itertools

from generic_pose.utils import to_var, to_np
from generic_pose.losses.distance_class import LossType, vertexDistanceLoss, vertexDistanceMetrics

def evaluateClassDistance(model, 
                          imgs, label_qs,
                          base_vertices,
                          loss_type = 'CrossEntropy',
                          falloff_angle = np.pi/9,
                          optimizer = None, 
                          calc_metrics = False,
                          kernal = None,
                          add_data = None,
                          ):
    #t = time.time()
    model.eval()
    #print("Get Loss:", time.time()-t)
    #t = time.time()
    if(optimizer is not None):
        optimizer.zero_grad()
    #print('pre_util')
    imgs = to_var(imgs)
    label_qs = to_var(label_qs)
    #print("Get Images:", time.time()-t)
    #t = time.time()
           
    #print("Calculate Samples:", time.time()-t)
    #t = time.time()
    pred_cls = model(imgs)
    loss = vertexDistanceLoss(pred_cls, label_qs, base_vertices, 
                              falloff_angle = falloff_angle,
                              loss_type = LossType[loss_type])
    #print("Calc Loss:", time.time()-t)
    #t = time.time()

    if(optimizer is not None):
        model.train()
        loss.backward()
        #loss.backward(retain_graph=retain_graph)
        #if(clip is not None):
        #    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

    #print("Backprop:", time.time()-t)
    #t = time.time()i
    results = {}
    results['loss'] = float(to_np(loss))

    if(calc_metrics):
        metrics = vertexDistanceMetrics(pred_cls, label_qs, base_vertices, 
                add_data = add_data, kernal = kernal)
        for k,v in metrics.items():
            if(type(v) == torch.Tensor):
                v = to_np(v)
            results[k] = np.mean(v) 
        
        results['mean_features'] = np.mean(np.abs(to_np(pred_cls)))
    #print("Display Metrics:", time.time()-t)
    #t = time.time()

    del loss 
    torch.cuda.empty_cache()
    #print("Cleanup:", time.time()-t)
    #t = time.time()

    return results



