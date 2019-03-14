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
        if(type(feature_network) not in [list, len]):
            self.origin_network = feature_network
            self.query_network = feature_network
        elif(len(feature_network) == 2):
            self.origin_network = feature_network[0]
            self.query_network = feature_network[1]
        else:
            raise ValueError('feature_network must be network or pair of networks')
        self.feature_network = self.query_network
        self.compare_network = compare_network

    def features(self, x):
        return self.queryFeatures(x)

    def originFeatures(self, x):
        x = self.origin_network(x)
        x = x.view(x.size(0), -1)
        return x

    def queryFeatures(self, x):
        x = self.query_network(x)
        x = x.view(x.size(0), -1)
        return x
 
    def forward(self, origin, query):
        origin = self.originFeatures(origin)
        query  = self.queryFeatures(query)

        return self.compare_network(origin, query)

def gen_pose_net(feature_type, compare_type, output_dim = 4, pretrained = True, fix_features = False, siamese_features = True):
    
    assert feature_type in feature_networks.keys(), 'Invalid feature type {}, Must be in {}'.format(feature_type, feature_networks.keys())
    feature_net, feature_size = feature_networks[feature_type](pretrained = pretrained)
    if(not siamese_features):
        feature_net = [feature_net, feature_networks[feature_type](pretrained = pretrained)[0]]

    assert compare_type in compare_networks.keys(), 'Invalid compare type {}, Must be in {}'.format(compare_type, compare_networks.keys())
    compare_net = compare_networks[compare_type](feature_size, output_dim)
    
    model = GenericPoseNet(feature_net, compare_net)
    
    if(fix_features):
       for p in model.origin_network.parameters():
           p.requires_grad=False
       if(not siamese_features):
           for p in model.query_network.parameters():
               p.requires_grad=False
    
    return model

def load_state_dict(model, weight_file):
    if torch.cuda.is_available():
        weights_dict = torch.load(weight_file)
    else: 
        weights_dict = torch.load(weight_file, map_location=lambda storage, loc: storage)
    if(True or model.origin_network != model.query_network):
        update_dict = {}
        for k, v in weights_dict.items(): 
            if('feature_network' in k):
                update_dict[k.replace('feature_network', 'origin_network')] = v
                update_dict[k.replace('feature_network', 'query_network')] = v
        weights_dict.update(update_dict)
    model.load_state_dict(weights_dict)

def add_mask_input(model):
    model.origin_network[0] = create_mask_input(model.origin_network[0])
    if(model.origin_network != model.query_network):
        model.query_network[0] = create_mask_input(model.query_network[0])

def create_mask_input(conv_in):
    conv_mask = nn.Conv2d(4, conv_in.out_channels, 
                          kernel_size=conv_in.kernel_size,
                          stride = conv_in.stride,
                          padding = conv_in.padding)
    state_dict = conv_in.state_dict()
    state_dict['weight'] = torch.cat([state_dict['weight'], 
        torch.randn((conv_in.out_channels,1,) + conv_in.kernel_size)], dim=1)
    conv_mask.load_state_dict(state_dict)
    return conv_mask


