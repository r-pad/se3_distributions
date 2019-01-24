# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""
import torch
import numpy as np
from quat_math import (quatDiff,
                       quatAngularDiff,
                       quaternion_multiply,
                       quaternion_inverse)

from functools import partial
from itertools import product

eps = 1e-7

def tensorAngularAllDiffs(label_qs, verts):
    return 2 * torch.acos(torch.abs(torch.transpose(torch.mm(verts, 
        torch.transpose(label_qs,0,1)),0,1)).clamp(max=1-eps))

def tensorAngularDiff(q1, q2):
    return 2 * torch.acos(torch.abs((q1*q2).sum(1)).clamp(max=1-eps))

def getGaussianKernal(verts, sigma=np.pi/36):
    kernal = torch.exp(-tensorAngularAllDiffs(verts,verts)**2/sigma)
    kernal = (kernal/kernal.sum(1)).transpose(0,1)
    return kernal


def pose2Viewpoint(q):
    delta_quat = np.array([-.5,-.5,-.5,.5])    
    q_flip = quaternion_multiply(delta_quat, q.copy())
    q_flip[2] *= -1
    return quaternion_inverse(q_flip)

def viewpoint2Pose(q):
    delta_quat = np.array([.5,.5,.5,.5])
    q_flip = quaternion_inverse(q.copy())
    q_flip[2] *= -1
    return quaternion_multiply(delta_quat, q_flip)

def quatDiffBatch(view_quats, base_quats):
    return np.array(list(map(quatDiff, view_quats, base_quats)))

def quatAngularDiffBatch(q1, q2):
    return 2*np.arccos(np.minimum(np.abs(np.sum(q1*q2, axis=1)), 1-eps))
    #return np.array(list(map(quatAngularDiff, view_quats, base_quats)))


def quatAngularDiffProd(view_quats, base_quats):
    #diff = partial(quatAngularDiff, view_quat)
    pairs = product(view_quats, base_quats)
    return np.array(list(map(lambda pair: quatAngularDiff(pair[0], pair[1]), pairs)))


def quatAngularDiffDot(q1, q2):
    return 2*np.arccos(np.minimum(np.abs(q1.dot(q2.T)), 1-eps))


def quaternionMultiplyBatch(q2, q1):
    return np.stack([ q2[:,0]*q1[:,3] + q2[:,1]*q1[:,2] - q2[:,2]*q1[:,1] + q2[:,3]*q1[:,0],
                      -q2[:,0]*q1[:,2] + q2[:,1]*q1[:,3] + q2[:,2]*q1[:,0] + q2[:,3]*q1[:,1],
                      q2[:,0]*q1[:,1] - q2[:,1]*q1[:,0] + q2[:,2]*q1[:,3] + q2[:,3]*q1[:,2],
                      q2[:,0]*q1[:,0] - q2[:,1]*q1[:,1] - q2[:,2]*q1[:,2] + q2[:,3]*q1[:,3]], dim=1)

def quaternioniAngularDifferenceBatch(q2, q1):
    return np.stack([ q2[:,0]*q1[:,3] + q2[:,1]*q1[:,2] - q2[:,2]*q1[:,1] + q2[:,3]*q1[:,0],
                      q2[:,0]*q1[:,2] + q2[:,1]*q1[:,3] + q2[:,2]*q1[:,0] + q2[:,3]*q1[:,1],
                      q2[:,0]*q1[:,1] - q2[:,1]*q1[:,0] + q2[:,2]*q1[:,3] + q2[:,3]*q1[:,2],
                      q2[:,0]*q1[:,0] + q2[:,1]*q1[:,1] + q2[:,2]*q1[:,2] + q2[:,3]*q1[:,3]], dim=1)


