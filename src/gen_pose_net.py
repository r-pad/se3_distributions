# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:42:27 2018

@author: bokorn
"""
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models.vgg as vgg
import torchvision.models.alexnet as alexnet
import torchvision.models.resnet as resnet

class GenPoseNet(nn.Module):

    def __init__(self, 
                 features_classification, features_classification_size, 
                 features_regression, features_regression_size, 
                 classification_output_dims=(360,360,360)):
        super(GenPoseNet, self).__init__()
        
        self.features_classification = features_classification
        
        self.features_regression = features_regression

        if(self.features_classification is not None):
            self.compare_classification = nn.Sequential(
                nn.Dropout(),
                nn.Linear(features_classification_size * 2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
            )
    
            self.dim0_linear = nn.Linear(4096, classification_output_dims[0])
            self.dim1_linear = nn.Linear(4096, classification_output_dims[1])
            self.dim2_linear = nn.Linear(4096, classification_output_dims[2])
        else:
            self.compare_classification = None
            self.dim0_linear = None
            self.dim1_linear = None
            self.dim2_linear = None
            
        if(self.features_regression is not None):
            self.compare_regression = nn.Sequential(
                nn.Dropout(),
                nn.Linear(features_regression_size * 2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4),
            )
        else:
            self.compare_regression = None
            
    def forwardClassification(self, origin, query):
        origin = self.featuresClassification(origin)
        query  = self.featuresClassification(query)
        return self.compareClassification(origin, query)

    def featuresClassification(self, x):
        if(self.features_classification is None):
            raise AssertionError('Classification feature network not set.')
        x = self.features_classification(x)       
        x = x.view(x.size(0), -1)
        return x

    def compareClassification(self, origin_features, query_features):
        if(self.compare_classification is None):
            raise AssertionError('Classification comparison network not set.')
        x = self.compare_classification(torch.cat((origin_features, query_features), dim=1))
        dim0 = self.dim0_linear(x)
        dim1 = self.dim1_linear(x)
        dim2 = self.dim2_linear(x)
        return dim0, dim1, dim2

    def forwardRegression(self, origin, query):
        origin = self.featuresRegression(origin)
        query  = self.featuresRegression(query)
        return self.compareRegression(origin, query)
        
    def featuresRegression(self, x):
        if(self.features_regression is None):
            raise AssertionError('Regression feature network not set.')
        x = self.features_regression(x)
        x = x.view(x.size(0), -1)
        return x

    def compareRegression(self, origin_features, query_features):
        if(self.compare_regression is None):
            raise AssertionError('Regression comparison network not set.')
        x = self.compare_regression(torch.cat((origin_features, query_features), dim=1))
        return x

    def forward(self, origin, query):
        quat = self.forwardRegression(origin, query)
        dim0, dim1, dim2 = self.forwardClassification(origin, query)
        return quat, dim0, dim1, dim2

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
    
def VGG16Features(batch_norm=True):
    features = vgg.make_layers(vgg.cfg['D'], batch_norm=batch_norm)
    return features
    
#
#def _resnet_make_layer(block, planes, blocks, inplanes, stride=1):
#    downsample = None
#    if stride != 1 or inplanes != planes * block.expansion:
#        downsample = nn.Sequential(
#            nn.Conv2d(inplanes, planes * block.expansion,
#                      kernel_size=1, stride=stride, bias=False),
#            nn.BatchNorm2d(planes * block.expansion),
#        )
#
#    layers = []
#    layers.append(block(inplanes, planes, stride, downsample))
#    inplanes = planes * block.expansion
#    for i in range(1, blocks):
#        layers.append(block(inplanes, planes))
#
#    return nn.Sequential(*layers), inplanes
#
#def ResNetFeatures(block, layers):
#    inplanes = 64
#    
#    conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                           bias=False)
#    bn1 = nn.BatchNorm2d(64)
#    relu = nn.ReLU(inplace=True)
#    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#    layer1, inplanes = _resnet_make_layer(block, 64, layers[0], inplanes=inplanes)
#    layer2, inplanes = _resnet_make_layer(block, 128, layers[1], inplanes=inplanes, stride=2)
#    layer3, inplanes = _resnet_make_layer(block, 256, layers[2], inplanes=inplanes, stride=2)
#    layer4, inplanes = _resnet_make_layer(block, 512, layers[3], inplanes=inplanes, stride=2)
#    avgpool = nn.AvgPool2d(7, stride=1)
#
#    features = nn.Sequential(
#        conv1,
#        bn1,
#        relu,
#        maxpool,
#        layer1,
#        layer2,
#        layer3,
#        layer4,
#        avgpool,
#    )
#
#    for m in features.modules():
#        if isinstance(m, nn.Conv2d):
#            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#            m.weight.data.normal_(0, math.sqrt(2. / n))
#        elif isinstance(m, nn.BatchNorm2d):
#            m.weight.data.fill_(1)
#            m.bias.data.zero_()
#
#    return features
#    

class ResNetFeatures(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetFeatures, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x
        
def gen_pose_net_alexnet(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    features_classification = AlexnetFeatures()
    features_regression = AlexnetFeatures()
    feature_size = 256 * 6 * 6
    model = GenPoseNet(features_classification, feature_size, 
                       features_regression, feature_size, **kwargs)
    if pretrained:
        
        model_dict = model.state_dict()

        pretrained_dict = model_zoo.load_url(alexnet.model_urls['alexnet'])
        update_dict = {}

        for k, v in pretrained_dict.items():
            k_classification = str(k).replace('features', 'features_classification')
            if k_classification in model_dict:
                update_dict[k_classification] = v
            k_regression = str(k).replace('features', 'features_regression')
            if k_regression in model_dict:
                update_dict[k_regression] = v
                
        model_dict.update(update_dict) 

        model.load_state_dict(model_dict)
        
    return model
    
    
def gen_pose_net_vgg16(pretrained=False, batch_norm=True, **kwargs):
    """VGG16 mopdel archetecture
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    features_classification = VGG16Features(batch_norm)
    features_regression = VGG16Features(batch_norm)
    feature_size = 512 * 7 * 7
    model = GenPoseNet(features_classification, feature_size, 
                       features_regression, feature_size, **kwargs)
    
    if pretrained:        
        model_dict = model.state_dict()

        pretrained_dict = model_zoo.load_url(vgg.model_urls['vgg16'])
        update_dict = {}

        for k, v in pretrained_dict.items():
            k_classification = str(k).replace('features', 'features_classification')
            if k_classification in model_dict:
                update_dict[k_classification] = v
            k_regression = str(k).replace('features', 'features_regression')
            if k_regression in model_dict:
                update_dict[k_regression] = v
                
        model_dict.update(update_dict) 

        model.load_state_dict(model_dict)
        
    return model
    
def gen_pose_net_resnet101(classification = True, regression = True, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if(classification):
        resnet_classification = resnet.resnet101(pretrained=pretrained)
        lst_classification = [elem for elem in resnet_classification.children()]
        features_classification = torch.nn.Sequential(*lst_classification[:-1])
        features_classification_size = lst_classification[-1].in_features
    else:
        features_classification = None
        features_classification_size = 0
        
    if(regression):
        resnet_regression = resnet.resnet101(pretrained=pretrained)
        lst_regression = [elem for elem in resnet_regression.children()]
        features_regression = torch.nn.Sequential(*lst_regression[:-1])
        features_regression_size = lst_classification[-1].in_features
    else:
        features_regression = None
        features_regression_size = 0
        
    model = GenPoseNet(features_classification, features_classification_size,
                       features_regression, features_regression_size, **kwargs)

    return model
    
def gen_pose_net_resnet50(classification = True, regression = True, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if(classification):
        resnet_classification = resnet.resnet50(pretrained=pretrained)
        lst_classification = [elem for elem in resnet_classification.children()]
        features_classification = torch.nn.Sequential(*lst_classification[:-1])
        features_classification_size = lst_classification[-1].in_features
    else:
        features_classification = None
        features_classification_size = 0
        
    if(regression):
        resnet_regression = resnet.resnet50(pretrained=pretrained)
        lst_regression = [elem for elem in resnet_regression.children()]
        features_regression = torch.nn.Sequential(*lst_regression[:-1])
        features_regression_size = lst_classification[-1].in_features
    else:
        features_regression = None
        features_regression_size = 0
        
    model = GenPoseNet(features_classification, features_classification_size,
                       features_regression, features_regression_size, **kwargs)

    return model