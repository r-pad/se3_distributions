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
from collections import OrderedDict

from se3_distributions.models.posecnn_mask import PoseCNNMask

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

#def simple_features(pretrain = False):
#    return

class LinearFeatureNet(nn.Module):
    def __init__(self, conv_network, conv_outsize, linear_layers = [2048, 2048]):
        super(LinearFeatureNet, self).__init__()
        self.conv_network = conv_network
        ls_prev = conv_outsize
        layers = []
        for j, ls in enumerate(linear_layers):
            layers.append(('fc{}'.format(j+1), torch.nn.Linear(ls_prev, ls)))
            layers.append(('relu{}'.format(j+1), torch.nn.ReLU(inplace=True)))
            ls_prev = ls
        
        self.linear = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
        
def linear_features(convnet_generator, linear_layers = [2048, 2048], **kwargs):
    conv_network, outsize = convnet_generator(**kwargs)
    feature_network = LinearFeatureNet(conv_network, outsize, 
                                       linear_layers = linear_layers)
    return feature_network, linear_layers[-1]

def posecnn_features(pretrained = True, add_conv4_features = False):
    feature_network = PoseCNNMask(checkpoint_filename ='/home/bokorn/pretrained/pose_cnn/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt', 
            add_conv4_features = add_conv4_features)
    if(add_conv4_features):
        feature_size = 512*28*28
    else:
        feature_size = 512*14*14

    return feature_network, feature_size
    
def posecnn_imgnet_features(pretrained = True, add_conv4_features = False):
    feature_network = PoseCNNMask(checkpoint_filename ='/home/bokorn/pretrained/distance/vgg_16.ckpt', 
            imagenet_weights=True, add_conv4_features = add_conv4_features)
    if(add_conv4_features):
        feature_size = 512*28*28
    else:
        feature_size = 512*14*14

    return feature_network, feature_size
    



feature_networks = {
                    'posecnn_imgnet':partial(posecnn_imgnet_features, add_conv4_features=False),
                    'posecnn_imgnet_add':partial(posecnn_imgnet_features, add_conv4_features=True),
                    'posecnn':partial(posecnn_features, add_conv4_features=False),
                    'posecnn_add':partial(posecnn_features, add_conv4_features=True),
                    'alexnet':alexnet_features,
                    'vgg16':vgg16_features,
                    'resnet152':partial(resnet_features, version=152),
                    'resnet101':partial(resnet_features, version=101),
                    'resnet50':partial(resnet_features, version=50),
                    'resnet34':partial(resnet_features, version=34),
                    'resnet18':partial(resnet_features, version=18),
                    'alexnet_fc1':partial(linear_features, 
                                          convnet_generator=alexnet_features,
                                          linear_layers=[2048]),
                    'alexnet_fc2':partial(linear_features, 
                                          convnet_generator=alexnet_features),
                    }
