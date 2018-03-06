# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:00:13 2018

@author: bokorn
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models.vgg as vgg
#import torchvision.models.alexnet as alexnet
import torchvision.models.resnet as resnet
from functools import partial

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

def AlexnetFeatures():
    features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        )
    return features
        
def alexnet_features(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    feature_network = AlexnetFeatures()
    feature_size = 256 * 6 * 6
    if pretrained:
        model_dict = feature_network.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        update_dict = {}
        for k, v in pretrained_dict.items():
            k_update = str(k).replace('features.', '')
            if k_update in model_dict:
                update_dict[k_update] = v
        model_dict.update(update_dict) 
        feature_network.load_state_dict(model_dict)

    return feature_network, feature_size

def vgg16_features(pretrained=False, **kwargs):
    """VGG16 mopdel archetecture
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    vgg16_model = vgg.vgg16_bn(pretrained=pretrained)
    feature_network = vgg16_model.features
    feature_size = 512 * 7 * 7

    return feature_network, feature_size

resnet_init = {152:resnet.resnet152,
               101:resnet.resnet101,
               50:resnet.resnet50,
               34:resnet.resnet34,
               18:resnet.resnet18}
    
def resnet_features(pretrained=False, version = 101, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    assert version in resnet_init.keys(), 'Invalid Resnet version {}, Must be in {}'.format(version, resnet_init.keys())
    
    resnet_model = resnet_init[version](pretrained=pretrained)
    features_lst = [elem for elem in resnet_model.children()]
    feature_network = torch.nn.Sequential(*features_lst[:-1])
    feature_size = features_lst[-1].in_features
        
    return feature_network, feature_size
    
feature_networks = {
                    'alexnet':alexnet_features,
                    'vgg16':vgg16_features,
                    'resnet152':partial(resnet_features, version=152),
                    'resnet101':partial(resnet_features, version=101),
                    'resnet50':partial(resnet_features, version=50),
                    'resnet34':partial(resnet_features, version=34),
                    'resnet18':partial(resnet_features, version=18),
                    }