from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random

from object_pose_utils.utils import to_np
from object_pose_utils.utils.bingham import iso_loss_calculation, duel_loss_calculation

class isoLikelihood(object):
        def __init__(self, mean_q, sig):
            self.mean_q = mean_q
            self.sig = sig.flatten()
            
        def __call__(self, quats):
            likelihoods = []
            for q in quats.unsqueeze(1):
                loss, lik = iso_loss_calculation(self.mean_q, torch.abs(self.sig), q)
                likelihoods.append(lik*2.0)
            return torch.stack(likelihoods)
        
class duelLikelihood(object):
        def __init__(self, mean_q, duel_q, z):
            self.mean_q = mean_q
            self.duel_q = duel_q
            self.z = z.flatten()
            
        def __call__(self, quats):
            likelihoods = []
            for q in quats.unsqueeze(1):
                loss, lik = duel_loss_calculation(self.mean_q, self.duel_q, 
                                                  -torch.abs(self.z), q)
                likelihoods.append(lik*2.0)
            return torch.stack(likelihoods)

class mLikelihood(object):
    def __init__(self, M, Z):
        self.M = M
        self.Z = Z
    def __call__(self, quats):
        return 2*bingham_likelihood(self.M, self.Z, quats.cuda())
        
def isoLoss(pred_mean, pred_sigma, true_r, return_all=False):
    losses = []
    likelihoods = []

    for q, sig, x in zip(pred_mean, pred_sigma, true_r.unsqueeze(1)):
        loss, lik = iso_loss_calculation(q, torch.abs(sig), x)
        losses.append(loss)
        likelihoods.append(lik)
    if(return_all):
        return torch.stack(losses), torch.stack(likelihoods)
    return torch.stack(losses).mean(), torch.stack(likelihoods).mean()

def duelLoss(pred_mean ,pred_duel, pred_z, true_r):
    losses = []
    likelihoods = []

    for qm, qd, z, x in zip(pred_mean, pred_duel, pred_z, true_r.unsqueeze(1)):
        loss, lik = duel_loss_calculation(qm, qd, -torch.abs(z), x)
        losses.append(loss)
        likelihoods.append(lik)
    
    return torch.stack(losses).mean(), torch.stack(likelihoods).mean()

def evaluateLikelihood(model, objs, quats,
                       pred_quats, features, 
                       optimizer = None,
                       retain_graph = False,
                       calc_metrics = False,
                       clip_value = 100.,
                       ):
    
    objs = objs.cuda() - 1 
    quats = quats.cuda()
    pred_quats = pred_quats.cuda()
    features = features.cuda()
    res = model(features, objs)

    if(type(res) is not tuple):
        pred_sigma = res[:,0,:]
        loss, lik = isoLoss(pred_quats, pred_sigma, quats)

    elif(len(res) == 2):
        pred_duel, pred_z = res
        pred_duel = pred_duel[:,0,:]
        pred_z = pred_z[:,0,:]
        loss, lik = duelLoss(pred_quats, pred_duel, pred_z, quats)

    if(not torch.isnan(loss) and optimizer is not None):
        model.train()
        loss.backward(retain_graph=retain_graph)
        #torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), clip_value)

        for p in model.parameters():
            if(torch.any(torch.isnan(p.grad.data)) or \
               torch.any(torch.isinf(p.grad.data))):
                p.grad.data = 0.

        optimizer.step()

    metrics = {}
    metrics['loss'] = float(to_np(loss))
    if(calc_metrics):
        metrics['lik'] = float(to_np(loss))

    return metrics  

