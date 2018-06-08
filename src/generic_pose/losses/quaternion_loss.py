# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:05:43 2017

@author: bokorn
"""
import numpy as np 

import torch
from torch.autograd import Variable

from generic_pose.utils.data_preprocessing import quatAngularDiff, quat2AxisAngle

def quaternionLoss(preds, labels, mean = True):
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
    #In case of zero quats
    qn_pad = qn + (qn==0).detach().float()
    
    preds = preds.div(qn_pad.unsqueeze(1).expand_as(preds))
    loss = 1 - torch.sum(torch.mul(preds, labels.float()), 1)**2
    if(mean):
        return torch.mean(loss)
    else:
        return loss

def axisLoss(preds, labels, mean = True):
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
    
    ln = torch.norm(labels[:,:3], p=2, dim=1)#.detach()
    ax_label = labels[:,:3].div(ln.unsqueeze(1).expand_as(labels[:,:3]))
    
    pn = torch.norm(preds[:,:3], p=2, dim=1)#.detach()
    #In case of zero quats
    pn_pad = pn + (pn==0).detach().float()
    
    ax_pred = preds[:,:3].div(pn_pad.unsqueeze(1).expand_as(preds[:,:3]))
    ax_pred *= torch.sign(preds[:,3]).unsqueeze(dim=1)
    #preds = preds.div(qn_pad.unsqueeze(1).expand_as(preds))
    
    loss = .5 - .5*torch.sum(torch.mul(ax_pred, ax_label.float()), 1)
    if(mean):
        return torch.mean(loss)
    else:
        return loss

def blendedLoss(preds, labels, min_angle = np.pi/4.0, max_angle = np.pi):
    th_true = quaternionAngles(labels)
    th_true = np.minimum(np.maximum(th_true, min_angle), max_angle);
    gamma = torch.from_numpy((th_true - min_angle)/(max_angle - min_angle)).float()
    if torch.cuda.is_available():
        gamma = gamma.cuda()
    gamma = Variable(gamma)
    loss = gamma * axisLoss(preds, labels, mean = False) + (1-gamma) * quaternionLoss(preds, labels, mean = False)
    return torch.mean(loss)

def clipQuatAngle(qs, max_angle=np.pi/4.0):
    clipped_q = qs.clone()
    cos_max = np.cos(max_angle/2.0)
    sin_max = np.sin(max_angle/2.0)
    cos_th = qs[:,3].div(torch.norm(qs, p=2, dim=1))
    for j, c_th in enumerate(cos_th):
        if (torch.abs(c_th) < cos_max).all():
            axis = qs[j, :3]
            if(c_th < 0).all():
                axis = -axis
            axis_norm = torch.norm(axis, p=2)
            axis = axis.div(axis_norm.expand_as(axis))
            clipped_q[j, :3] = axis*sin_max
            clipped_q[j, 3] = cos_max
    return clipped_q

def axisAngle2Quat(axis_angle, max_angle=np.pi/4.0):
    q = torch.zeros(axis_angle.size()).float()
    if torch.cuda.is_available():
        q = q.cuda()
    q = torch.autograd.Variable(q)
    cos_th = torch.cos(0.5*torch.clamp(axis_angle[:,3], min=-max_angle, max=max_angle))
    sin_th = torch.sin(0.5*torch.clamp(axis_angle[:,3], min=-max_angle, max=max_angle))
    axis_norms = torch.norm(axis_angle[:,:3], p=2, dim=1)
    q[:,:3] = axis_angle[:,:3] * (sin_th/axis_norms).unsqueeze(1).expand_as(axis_angle[:,:3])
    q[:,3] = cos_th
    return q

def tensor2Angle(q):
    q *= torch.sign(q[:,3]).unsqueeze(dim=1)
    q[:,3] = torch.clamp(q[:,3],max=1.0)

    angle = 2*torch.acos(q[:,3])
    axis = q[:,:3].div(angle.unsqueeze(1).expand_as(q[:,:3]))
    
    return axis, angle
    
def quaternionError(preds, labels):
    batch_size = preds.size(0)
    est_quats = preds.data.cpu().numpy()
    true_quats = labels.data.cpu().numpy()
    error = np.zeros(batch_size)
    for inst_id in range(batch_size):
        qn = np.linalg.norm(est_quats[inst_id])
        est_q = est_quats[inst_id]/qn
        est_q *= np.sign(est_q[3])
        true_q = true_quats[inst_id]
        true_q *= np.sign(true_q[3])
        diff = quatAngularDiff(est_q, true_q)
        if(diff > np.pi):
            diff = 2.0*np.pi - diff
        error[inst_id] = diff
    return error

def axisError(preds, labels):
    batch_size = preds.size(0)
    est_quats = preds.data.cpu().numpy()
    true_quats = labels.data.cpu().numpy()
    error = np.zeros(batch_size)
    for inst_id in range(batch_size):
        qn = np.linalg.norm(est_quats[inst_id])
        est_q = est_quats[inst_id]/qn
        est_q *= np.sign(est_q[3])
        true_q = true_quats[inst_id]
        true_q *= np.sign(true_q[3])
        
        est_axis = quat2AxisAngle(est_q)[0]
        true_axis = quat2AxisAngle(true_q)[0]
        error[inst_id] = np.arccos(np.dot(est_axis,true_axis))
    return error
    
def quaternionAngles(labels):
    batch_size = labels.size(0)
    quats = labels.data.cpu().numpy()
    angles = np.zeros(batch_size)
    for inst_id in range(batch_size):
        q = quats[inst_id]
        q *= np.sign(q[3])
        th = quat2AxisAngle(q)[1]
        if(th > np.pi):
            th = 2.0*np.pi - th
        angles[inst_id] = th
    return angles

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
        angle = np.abs(diff - 2*np.pi*(diff > np.pi))
    else:
        angle = None
        
    return loss, angle

def quaternionInverse(q):
    q_inv = torch.zeros(q.size()).float()
    if torch.cuda.is_available():
        q_inv = q_inv.cuda()
    q_inv = torch.autograd.Variable(q_inv)
    
    q_inv[:,:3] = -q[:,:3]
    q_inv[:,3] = q[:,3]
    return q_inv

def quaternionMultiply(q2, q1):
    
    return torch.stack([ q2[:,0]*q1[:,3] + q2[:,1]*q1[:,2] - q2[:,2]*q1[:,1] + q2[:,3]*q1[:,0],
                        -q2[:,0]*q1[:,2] + q2[:,1]*q1[:,3] + q2[:,2]*q1[:,0] + q2[:,3]*q1[:,1],
                         q2[:,0]*q1[:,1] - q2[:,1]*q1[:,0] + q2[:,2]*q1[:,3] + q2[:,3]*q1[:,2],
                        -q2[:,0]*q1[:,0] - q2[:,1]*q1[:,1] - q2[:,2]*q1[:,2] + q2[:,3]*q1[:,3]], dim=1)