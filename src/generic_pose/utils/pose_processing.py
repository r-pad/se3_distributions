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
                       quaternion_inverse,
                       quaternion_about_axis)

from functools import partial
from itertools import product

eps = 1e-7

def tensorAngularAllDiffs(label_qs, verts):
    return 2 * torch.acos(torch.abs(torch.transpose(torch.mm(verts, 
        torch.transpose(label_qs,0,1)),0,1)).clamp(max=1-eps))

def tensorSignedAngularDiff(q1, q2):
    return 2 * torch.acos(torch.mm(q1,q2.t()).clamp(min=eps-1, max=1-eps))

def tensorAngularDiff(q1, q2):
    return 2 * torch.acos(torch.abs((q1*q2).sum(1)).clamp(max=1-eps))

def getGaussianKernal(verts, sigma=np.pi/36):
    kernal = torch.exp(-tensorAngularAllDiffs(verts,verts)**2/sigma)
    kernal = (kernal/kernal.sum(1)).transpose(0,1)
    return kernal

def getSignedGaussianKernal(pts, verts, sigma=np.sqrt(2)*np.pi/4):
    d_pos = tensorSignedAngularDiff(pts, verts)
    d_neg = tensorSignedAngularDiff(pts, -verts)
    d = torch.min(d_pos, d_neg)
    sign = 1-2*(d_neg < d_pos).float()
    kernal = sign * torch.exp(-d**2/sigma**2)
    return kernal

def meanShiftStep(pts, verts, weights, sigma=np.sqrt(2)*np.pi/4):
    K = weights.t() * getSignedGaussianKernal(pts, verts, sigma)
    new_pts = torch.mm(K, verts)
    new_pts /= torch.norm(new_pts,dim=1).unsqueeze(1)
    return new_pts

def meanShift(pts, verts, weights, sigma=np.sqrt(2)*np.pi/4, eps_term = 1e-5, max_iter = 100):
    for _ in range(max_iter):
        new_pts = meanShiftStep(pts, verts, weights, sigma)
        if(torch.max(torch.abs(pts-new_pts)) < eps_term):
            break;
    return new_pts
            
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


def tensorMultiplyBatch(q1,q2):
    return torch.stack([ q1[:,0]*q2[:,3] + q1[:,1]*q2[:,2] - q1[:,2]*q2[:,1] + q1[:,3]*q2[:,0],
                        -q1[:,0]*q2[:,2] + q1[:,1]*q2[:,3] + q1[:,2]*q2[:,0] + q1[:,3]*q2[:,1],
                         q1[:,0]*q2[:,1] - q1[:,1]*q2[:,0] + q1[:,2]*q2[:,3] + q1[:,3]*q2[:,2],
                        -q1[:,0]*q2[:,0] - q1[:,1]*q2[:,1] - q1[:,2]*q2[:,2] + q1[:,3]*q2[:,3]], dim=1)

    
def continuousSymmetricAngularDistance(q1, q2, axis_sym):
    num_q = q1.shape[0];
   
    xi_sym = torch.cat((axis_sym, torch.tensor([0.]))).unsqueeze(0)
    
    q1_inv = q1*torch.tensor([-1.,-1.,-1.,1.])
    xi1 = tensorMultiplyBatch(tensorMultiplyBatch(q1,xi_sym),q1_inv)

    q2_inv = q2*torch.tensor([-1.,-1.,-1.,1.])
    xi2 = tensorMultiplyBatch(tensorMultiplyBatch(q2,xi_sym),q2_inv)
    
    dot = (xi1 * xi2).sum(dim=1)
    return torch.acos(dot.clamp(min = eps-1, max=1-eps))

def symmetricAngularDistance(q1, q2, axes_of_sym, angles_of_sym):
    if(len(axes_of_sym) == 0):
        return tensorAngularDiff(q1, q2)
    axis = axes_of_sym[0]
    angles = angles_of_sym[0]
    dist = torch.tensor(np.inf)
    for th in angles:
        if(np.isfinite(th)):
            q_sym = torch.tensor(quaternion_about_axis(th, axis)).float().unsqueeze(0)
            q2_rot = tensorMultiplyBatch(q2, q_sym)
            dist = torch.min(dist, symmetricAngularDistance(q1, q2_rot, axes_of_sym[1:], angles_of_sym[1:]))
        else:
            return continuousSymmetricAngularDistance(q1,q2,torch.tensor(axis).float())
    return dist

                 
                
