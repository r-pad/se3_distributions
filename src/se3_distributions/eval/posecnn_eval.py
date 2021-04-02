# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn 
"""
import os
import numpy as np
import scipy.io as sio

from se3_distributions.eval.pose_error import *
from quat_math import quaternion_matrix

default_extend_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/ycb_extents.txt')

def getYCBThresholds(extent_file = default_extend_file):
    assert os.path.exists(extent_file), \
	    'Path does not exist: {}'.format(extent_file)

    num_classes = 22
    extents = np.zeros((num_classes, 3), dtype=np.float32)
    extents[1:,:] = np.loadtxt(extent_file)
    threshold = np.zeros((num_classes,), dtype=np.float32)
            
    for i in range(num_classes):
        threshold[i] = 0.1 * np.linalg.norm(extents[i, :])

    return threshold 

def evaluatePoses(dataset, quat_pred, threshold):
    points = dataset.getObjectPoints()
    cls = dataset.getObjectName()
    use_sym = cls == '024_bowl' or cls == '036_wood_block' or cls == '061_foam_brick'
    quat_true = dataset.quats 
    
    errors = []
    for q_true, q_pred in zip(quat_true, quat_pred):
        errors.append(evaluateQuat(q_true, q_pred, points, use_sym = use_sym))
    return np.mean(errors < threshold), errors

def evaluateQuat(q_true, q_pred, points, use_sym = True, t_true = np.zeros(3), t_pred = np.zeros(3)):
    R_true = quaternion_matrix(q_true)[:3,:3] 
    R_pred = quaternion_matrix(q_pred)[:3,:3] 
    if use_sym:
        error = adi(R_pred, t_pred, R_true, t_true, points)
    else:
        error = add(R_pred, t_pred, R_true, t_true, points)

    return error
