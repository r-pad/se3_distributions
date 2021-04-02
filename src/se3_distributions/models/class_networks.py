# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:42:27 2018

@author: bokorn
"""

import torch
import torch.nn as nn
from se3_distributions.models.feature_networks import feature_networks
from se3_distributions.models.compare_networks import class_networks

class ClassPoseNet(nn.Module):
    def __init__(self, feature_network, output_network):
        super(ClassPoseNet, self).__init__()
        self.feature_network = feature_network
        self.output_network = output_network


    def features(self, x):
        x = self.feature_network(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.features(x)
        return self.output_network(x)

def gen_class_net(feature_type, output_type, output_dim = 4, pretrained = True, fix_features = False):
    
    assert feature_type in feature_networks.keys(), 'Invalid feature type {}, Must be in {}'.format(feature_type, feature_networks.keys())
    feature_net, feature_size = feature_networks[feature_type](pretrained = pretrained)

    assert output_type in class_networks.keys(), 'Invalid output type {}, Must be in {}'.format(output_type, benchmark_loaderss_networks.keys())
    output_net = class_networks[output_type](feature_size, output_dim)
    
    model = ClassPoseNet(feature_net, output_net)
    
    if(fix_features):
       for p in model.feature_network.parameters():
           p.requires_grad=False
    
    return model

