"""
Multi-class Geometric Viewpoint Aware Loss
A PyTorch implmentation of the geometric-aware softmax view loss as described in
RenderForCNN (link: https://arxiv.org/pdf/1505.05641.pdf)
Caffe implmentation:
https://github.com/charlesq34/caffe-render-for-cnn/blob/view_prediction/
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from data_preprocessing import uniform2Quat, quatAngularDiff

class ViewpointLoss(nn.Module):
    def __init__(self, mean = True):
        super(ViewpointLoss, self).__init__()
        self.mean = mean

    def forward(self, preds, labels):
        """
        :param preds:   Angle predictions (batch_size, 360)
        :param targets: Angle labels (batch_size, 360)
        :return: Loss. Loss is a variable which may have a backward pass performed.
        Apply softmax over the preds, and then apply geometrics loss
        """
        # Set absolute minimum for numerical stability (assuming float16 - 6x10^-5)
        # preds = F.softmax(preds.float())
        labels = labels.float()

        if self.mean:
            loss = -(labels * F.log_softmax(preds)).mean(1).sum()
        else:
            loss = -(labels * F.log_softmax(preds)).sum(1).sum()
            
        loss = loss.sum()

        return loss
        
def viewpointAccuracy(preds, labels):
    acc = 180 - ((preds.max(1)[1] - labels.max(1)[1]).abs() - 180).abs().data.cpu().numpy()
    return acc.sum() / acc.shape[0]
    #return acc
    
def denseViewpointError(preds, labels, num_bins = (100, 100, 50)):
    num_bins_full = num_bins[:2] + (2*num_bins[2],)

    batch_size = preds.size(0)
    
    err = 0
    for inst_id in range(batch_size):
        pred_max_idx = np.unravel_index(np.argmax(preds[inst_id, :].data.cpu().numpy()), num_bins)
        label_max_idx = np.unravel_index(np.argmax(labels[inst_id, :].data.cpu().numpy()), num_bins)
        pred_u = (np.array(pred_max_idx) + 0.5)/np.array(num_bins_full)
        label_u = (np.array(label_max_idx) + 0.5)/np.array(num_bins_full)
        pred_q = uniform2Quat(pred_u)
        label_q = uniform2Quat(label_u)
        err += quatAngularDiff(pred_q, label_q)
        
    return err/batch_size