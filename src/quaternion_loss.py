# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:05:43 2017

@author: bokorn
"""
import torch

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