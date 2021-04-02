# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:15:46 2018

@author: bokorn
"""

import torch
import torch.nn as nn
from se3_distributions.models.pose_networks import GenericPoseNet
from se3_distributions.models.feature_networks import feature_networks
from se3_distributions.models.compare_networks import compare_networks

class FeatureDynamicsNetwork(GenericPoseNet):
    def __init__(self, feature_network, compare_network, 
                 dynamics_network = None, 
                 feature_size = 2048, 
                 orientation_size = 4):
        super(FeatureDynamicsNetwork, self).__init__(feature_network=feature_network, 
                                                     compare_network=compare_network)
        if(dynamics_network is not None):
            self.dynamics_network = dynamics_network
        else:
            self.dynamics_network = nn.Sequential(
                                                  nn.Dropout(),
                                                  nn.Linear(feature_size + orientation_size, 4096),
                                                  nn.ReLU(inplace=True),
                                                  nn.Dropout(),
                                                  nn.Linear(4096, 4096),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(4096, feature_size)
                                                  )

    def dynamics(self, features, orientation):
        features_prime = self.dynamics_network(torch.cat((features, orientation), dim=1))
        return features_prime
        
def gen_dyn_net(feature_type, compare_type, output_dim = 4, pretrained = True, fix_features = False):
    
    assert feature_type in feature_networks.keys(), 'Invalid feature type {}, Must be in {}'.format(feature_type, feature_networks.keys())
    feature_net, feature_size = feature_networks[feature_type](pretrained = pretrained)
    
    assert compare_type in compare_networks.keys(), 'Invalid compare type {}, Must be in {}'.format(compare_type, compare_networks.keys())
    compare_net = compare_networks[compare_type](feature_size, output_dim)
    
    model = FeatureDynamicsNetwork(feature_net, compare_net, 
                                   feature_size=feature_size,
                                   orientation_size=output_dim,)
    if(fix_features):
       for p in model.feature_network.parameters():
           p.requires_grad=False
    
    return model