# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:42:27 2018

@author: bokorn
"""

import torch
import torch.nn as nn
from generic_pose.models.feature_networks import feature_networks
from generic_pose.models.compare_networks import compare_networks

class GenericPoseNet(nn.Module):
    def __init__(self, feature_network, compare_network):
        super(GenericPoseNet, self).__init__()
        
        self.feature_network = feature_network
        self.compare_network = compare_network

    def features(self, x):
        x = self.feature_network(x)
        x = x.view(x.size(0), -1)
        return x
        
    def forward(self, origin, query):
        origin = self.features(origin)
        query  = self.features(query)

        return self.compare_network(origin, query)

def gen_pose_net(feature_type, compare_type, output_dim = 4, pretrained = True):
    
    assert feature_type in feature_networks.keys(), 'Invalid feature type {}, Must be in {}'.format(feature_type, feature_networks.keys())
    feature_net, feature_size = feature_networks[feature_type](pretrained = pretrained)
    
    assert compare_type in compare_networks.keys(), 'Invalid compare type {}, Must be in {}'.format(compare_type, compare_networks.keys())
    compare_net = compare_networks[compare_type](feature_size, output_dim)
    
    model = GenericPoseNet(feature_net, compare_net)
    
    return model