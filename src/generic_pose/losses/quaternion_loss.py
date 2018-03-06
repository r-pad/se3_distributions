# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:05:43 2017

@author: bokorn
"""
import torch
import numpy as np 

from generic_pose.utils.data_preprocessing import quatAngularDiff

def quaternionLoss(preds, labels):
        """
        :param preds: quaternion predictions (batch_size, 1)
        :param labels: label quaternions labels (batch_size, 1)
        :return: Loss. Loss is a variable which may have a backward pass performed.
        """
        labels = labels.float()
        labels *= torch.sign(labels[:,3]).unsqueeze(dim=1)
        preds = preds.float()
        loss = torch.zeros(1)

        if torch.cuda.is_available():
            loss = loss.cuda()
        loss = torch.autograd.Variable(loss)
        
        qn = torch.norm(preds, p=2, dim=1)#.detach()
        #print(qn)
        preds = preds.div(qn.unsqueeze(1).expand_as(preds))
        loss = torch.mean(1 - torch.sum(torch.mul(preds, labels.float()), 1)**2)
        return loss
        
def quaternionError(preds, labels):
    batch_size = preds.size(0)
    est_quats = preds.data.cpu().numpy()
    true_quats = labels.data.cpu().numpy()
    acc = 0
    for inst_id in range(batch_size):
        est_q = est_quats[inst_id]/np.linalg.norm(est_quats[inst_id])
        est_q *= np.sign(est_q[3])
        true_q = true_quats[inst_id]
        true_q *= np.sign(true_q[3])
        diff = quatAngularDiff(est_q, true_q)
        if(diff > np.pi):
            diff = 2.0*np.pi - diff
        acc += diff

    return acc/batch_size

def loopConsistencyLoss(loop_transforms, calc_angle=False):
    n = loop_transforms[0].size(0)
    q_loop = torch.zeros(n, 4).float()
    q_loop[:,3] = 1.0
    
    loss = torch.zeros(1)

    if torch.cuda.is_available():
        q_loop = q_loop.cuda()
        loss = loss.cuda()
        
    q_loop = torch.autograd.Variable(q_loop)
    loss = torch.autograd.Variable(loss)
 
    for q in loop_transforms:
        qn = torch.norm(q, p=2, dim=1)#.detach()
        q = q.div(qn.unsqueeze(1).expand_as(q)).float()

        q_loop = quaternionMultiply(q_loop, q)
        
    loss = torch.mean(1 - q_loop[:,3]**2)
    if(calc_angle):
        diff = 2*np.arccos(q_loop[:,3].data.cpu().numpy())
        angle = np.mean(np.abs(diff - 2*np.pi*(diff > np.pi)))        
    else:
        angle = None
        
    return loss, angle

def quaternionMultiply(q1, q2):
    
    return torch.stack([ q2[:,0]*q1[:,3] + q2[:,1]*q1[:,2] - q2[:,2]*q1[:,1] + q2[:,3]*q1[:,0],
                        -q2[:,0]*q1[:,2] + q2[:,1]*q1[:,3] + q2[:,2]*q1[:,0] + q2[:,3]*q1[:,1],
                         q2[:,0]*q1[:,1] - q2[:,1]*q1[:,0] + q2[:,2]*q1[:,3] + q2[:,3]*q1[:,2],
                        -q2[:,0]*q1[:,0] - q2[:,1]*q1[:,1] - q2[:,2]*q1[:,2] + q2[:,3]*q1[:,3]], dim=1)