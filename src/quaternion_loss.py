# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:05:43 2017

@author: bokorn
"""
import torch
import numpy as np 

from data_preprocessing import quatAngularDiff

def quaternionLoss(preds, labels):
        """
        :param preds: quaternion predictions (batch_size, 1)
        :param labels: label quaternions labels (batch_size, 1)
        :return: Loss. Loss is a variable which may have a backward pass performed.
        """
        labels = labels.float()
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
        acc += quatAngularDiff(est_q, true_quats[inst_id])

    return acc/batch_size