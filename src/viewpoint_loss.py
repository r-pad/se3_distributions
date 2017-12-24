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

class ViewpointLoss(nn.Module):
    def __init__(self, class_period = 360, mean = True):
        super(ViewpointLoss, self).__init__()
        self.class_period = class_period
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
        batch_size = preds.size(0)
        loss = torch.zeros(1)

        if torch.cuda.is_available():
            loss = loss.cuda()
        loss = torch.autograd.Variable(loss)
        
        for inst_id in range(batch_size):
            if self.mean:
                loss -= (labels[inst_id, :] * F.log_softmax(preds[inst_id, :])).mean()
            else:
                loss -= (labels[inst_id, :] * F.log_softmax(preds[inst_id, :])).sum()

#        if self.mean:
#            loss = -(labels * F.log_softmax(preds)).mean(1).sum()
#        else:
#            loss = -(labels * F.log_softmax(preds)).sum(1).sum()
#        loss = loss.sum()
        return loss
        
def viewpointAccuracy(preds, labels):
    acc = 180 - ((preds.max(1)[1] - labels.max(1)[1]).abs() - 180).abs().data.cpu().numpy()
    return acc.sum() / acc.shape[0]
    #return acc