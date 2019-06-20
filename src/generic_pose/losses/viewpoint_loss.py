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
        return viewpointLoss(preds, labels, self.mean)
   
def viewpointLoss(preds, labels, mean = True):
    labels = labels.float()
    if mean:
        loss = -(labels * F.log_softmax(preds)).mean(1).sum()
    else:
        loss = -(labels * F.log_softmax(preds)).sum(1).sum()
    loss = loss.sum()
    return loss

def dotLoss(preds, labels):
    labels = labels.float()
    return -(labels * F.softmax(preds, dim=1)).sum()
    
def viewpointAccuracy(preds, labels):
    acc = 180 - ((preds.max(1)[1] - labels.max(1)[1]).abs() - 180).abs().data.cpu().numpy()
    return acc.sum() / acc.shape[0]
    #return acc

def denseViewpointError(preds, labels, num_bins = (50, 50, 25), filter_sigma = 1.0):
    batch_size = preds.size(0)
    err = 0
    idx_err = 0
    for inst_id in range(batch_size):
        vals = preds[inst_id, :].data.cpu().numpy().reshape(num_bins)
        filtered_preds = scipy.ndimage.filters.gaussian_filter(vals, sigma=filter_sigma, mode='wrap')
        pred_max_idx = np.unravel_index(np.argmax(filtered_preds), num_bins)
        label_max_idx = np.unravel_index(np.argmax(labels[inst_id, :].data.cpu().numpy()), num_bins)
        diff = indexAngularDiff(pred_max_idx, label_max_idx, num_bins)
        if(diff > np.pi):
            diff = 2.0*np.pi - diff
        err += diff
        idx_err += np.linalg.norm(np.array(pred_max_idx) - np.array(label_max_idx))
    
    return err/batch_size, idx_err/batch_si
