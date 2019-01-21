# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:00:13 2018

@author: bokorn
"""

import torch
import torch.nn as nn
from tensorflow.python import pywrap_tensorflow                                                            

from collections import OrderedDict

def torch2tfKey(k):
    return k.replace('.','/').replace('weight', 'weights').replace('bias','biases')
    #return k.replace('.','/').replace('weight', 'weights').replace('bias','biases').replace('/net/1', '').replace('/net/0', '')

def loadCheckpointDict(model, checkpoint_reader):
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k in model_dict.keys():
        if('weight' in k):
            pretrained_dict[k] = torch.from_numpy(checkpoint_reader.get_tensor(torch2tfKey(k))).permute(3, 2, 0, 1)
        else:
            pretrained_dict[k] = torch.from_numpy(checkpoint_reader.get_tensor(torch2tfKey(k)))
    model.load_state_dict(pretrained_dict)

class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ZeroPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)

class ConvTranspose2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        ka = -stride // 2
        kb = ka + 1 if stride % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, bias=False),
            nn.ZeroPad2d((ka,kb,ka,kb)),
        )
    def forward(self, x):
        return self.net(x)

def getPadingLayer(kernel_size, padding_layer=nn.ZeroPad2d, transpose = False):
    ka = kernel_size // 2
    kb = ka - 1 if kernel_size % 2 == 0 else ka
    if(transpose):
        ka*=-1
        kb=-kb-1
    return padding_layer((ka,kb,ka,kb))

class PoseCNNMask(nn.Module):
    def __init__(self, checkpoint_filename):
        super(PoseCNNMask, self).__init__()

        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_filename)
        if(False):
            self.features_conv4_3 = nn.Sequential(OrderedDict([ 
                ('conv1_1', Conv2dSame(3, 64, kernel_size=3)),
                ('relu1_1', nn.ReLU(inplace=True)),
                ('conv1_2', Conv2dSame(64, 64, kernel_size=3)),
                ('relu1_2', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(kernel_size=2)),
                ('conv2_1', Conv2dSame(64, 128, kernel_size=3)),
                ('relu2_1', nn.ReLU(inplace=True)),
                ('conv2_2', Conv2dSame(128, 128, kernel_size=3)),
                ('relu2_2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(kernel_size=2)),
                ('conv3_1', Conv2dSame(128, 256, kernel_size=3)),
                ('relu3_1', nn.ReLU(inplace=True)),
                ('conv3_2', Conv2dSame(256, 256, kernel_size=3)),
                ('relu3_2', nn.ReLU(inplace=True)),
                ('conv3_3', Conv2dSame(256, 256, kernel_size=3)),
                ('relu3_3', nn.ReLU(inplace=True)),
                ('pool3', nn.MaxPool2d(kernel_size=2)),
                ('conv4_1', Conv2dSame(256, 512, kernel_size=3)),
                ('relu4_1', nn.ReLU(inplace=True)),
                ('conv4_2', Conv2dSame(512, 512, kernel_size=3)),
                ('relu4_2', nn.ReLU(inplace=True)),
                ('conv4_3', Conv2dSame(512, 512, kernel_size=3)),
                ('relu4_3', nn.ReLU(inplace=True)),
                ]))
            loadCheckpointDict(self.features_conv4_3, reader) 

            self.features_conv5_3 = nn.Sequential(OrderedDict([ 
                ('pool4', nn.MaxPool2d(kernel_size=2)),
                ('conv5_1', Conv2dSame(512, 512, kernel_size=3)),
                ('relu5_1', nn.ReLU(inplace=True)),
                ('conv5_2', Conv2dSame(512, 512, kernel_size=3)),
                ('relu5_2', nn.ReLU(inplace=True)),
                ('conv5_3', Conv2dSame(512, 512, kernel_size=3)),
                ('relu5_3', nn.ReLU(inplace=True)),
                ]))
            loadCheckpointDict(self.features_conv5_3, reader) 
            
            self.features_upscore_conv5 = nn.Sequential(OrderedDict([ 
                ('score_conv5', Conv2dSame(512, 64, kernel_size=1)),
                ('relu_s5', nn.ReLU(inplace=True)),
                ('upscore_conv5', ConvTranspose2dSame(64, 64, kernel_size=4, stride=2)),
                ]))
            loadCheckpointDict(self.features_upscore_conv5, reader) 
                    
            self.features_score_conv4 = nn.Sequential(OrderedDict([ 
                ('score_conv4', Conv2dSame(512, 64, kernel_size=1)),
                ('relu_s4', nn.ReLU(inplace=True)),
                ]))
            loadCheckpointDict(self.features_score_conv4, reader) 

            self.features_prob = nn.Sequential(OrderedDict([ 
                #('dropout', nn.Dropout()),
                ('upscore', ConvTranspose2dSame(64, 64, kernel_size=16, stride=8)),
                ('score', Conv2dSame(64, 22, kernel_size=1)),
                ('relu_score', nn.ReLU(inplace=True)),
                ('prob', nn.LogSoftmax(dim=1))
                ]))
            loadCheckpointDict(self.features_prob, reader) 

        else:
            self.features_conv4_3 = nn.Sequential(OrderedDict([ 
                ('pad1_1', getPadingLayer(kernel_size=3)),
                ('conv1_1', nn.Conv2d(3, 64, kernel_size=3)),
                ('relu1_1', nn.ReLU(inplace=True)),
                ('pad1_2', getPadingLayer(kernel_size=3)),
                ('conv1_2', nn.Conv2d(64, 64, kernel_size=3)),
                ('relu1_2', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(kernel_size=2)),
                ('pad2_1', getPadingLayer(kernel_size=3)),
                ('conv2_1', nn.Conv2d(64, 128, kernel_size=3)),
                ('relu2_1', nn.ReLU(inplace=True)),
                ('pad2_2', getPadingLayer(kernel_size=3)),
                ('conv2_2', nn.Conv2d(128, 128, kernel_size=3)),
                ('relu2_2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(kernel_size=2)),
                ('pad3_1', getPadingLayer(kernel_size=3)),
                ('conv3_1', nn.Conv2d(128, 256, kernel_size=3)),
                ('relu3_1', nn.ReLU(inplace=True)),
                ('pad3_2', getPadingLayer(kernel_size=3)),
                ('conv3_2', nn.Conv2d(256, 256, kernel_size=3)),
                ('relu3_2', nn.ReLU(inplace=True)),
                ('pad3_3', getPadingLayer(kernel_size=3)),
                ('conv3_3', nn.Conv2d(256, 256, kernel_size=3)),
                ('relu3_3', nn.ReLU(inplace=True)),
                ('pool3', nn.MaxPool2d(kernel_size=2)),
                ('pad4_1', getPadingLayer(kernel_size=3)),
                ('conv4_1', nn.Conv2d(256, 512, kernel_size=3)),
                ('relu4_1', nn.ReLU(inplace=True)),
                ('pad4_2', getPadingLayer(kernel_size=3)),
                ('conv4_2', nn.Conv2d(512, 512, kernel_size=3)),
                ('relu4_2', nn.ReLU(inplace=True)),
                ('pad4_3', getPadingLayer(kernel_size=3)),
                ('conv4_3', nn.Conv2d(512, 512, kernel_size=3)),
                ('relu4_3', nn.ReLU(inplace=True)),
                ]))
            loadCheckpointDict(self.features_conv4_3, reader) 

            self.features_conv5_3 = nn.Sequential(OrderedDict([ 
                ('pool4', nn.MaxPool2d(kernel_size=2)),
                ('pad5_1', getPadingLayer(kernel_size=3)),
                ('conv5_1', nn.Conv2d(512, 512, kernel_size=3)),
                ('relu5_1', nn.ReLU(inplace=True)),
                ('pad5_2', getPadingLayer(kernel_size=3)),
                ('conv5_2', nn.Conv2d(512, 512, kernel_size=3)),
                ('relu5_2', nn.ReLU(inplace=True)),
                ('pad5_3', getPadingLayer(kernel_size=3)),
                ('conv5_3', nn.Conv2d(512, 512, kernel_size=3)),
                ('relu5_3', nn.ReLU(inplace=True)),
                ]))
            loadCheckpointDict(self.features_conv5_3, reader) 
            
            self.features_upscore_conv5 = nn.Sequential(OrderedDict([ 
                ('pad_s5', getPadingLayer(kernel_size=1)),
                ('score_conv5', nn.Conv2d(512, 64, kernel_size=1)),
                ('relu_s5', nn.ReLU(inplace=True)),
                ('upscore_conv5', nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, bias=False)),
                ('unpad_u5', getPadingLayer(kernel_size=2, transpose=True)),
                ]))
            loadCheckpointDict(self.features_upscore_conv5, reader) 
                    
            self.features_score_conv4 = nn.Sequential(OrderedDict([ 
                ('pad_s4', getPadingLayer(kernel_size=1)),
                ('score_conv4', nn.Conv2d(512, 64, kernel_size=1)),
                ('relu_s4', nn.ReLU(inplace=True)),
                ]))
            loadCheckpointDict(self.features_score_conv4, reader) 

            self.features_prob = nn.Sequential(OrderedDict([ 
                #('dropout', nn.Dropout()),
                ('upscore', nn.ConvTranspose2d(64, 64, kernel_size=16, stride=8, bias=False)),
                ('unpad_upscore', getPadingLayer(kernel_size=8, transpose = True)),
                ('pad_score', getPadingLayer(kernel_size=1)),
                ('score', nn.Conv2d(64, 22, kernel_size=1)),
                ('relu_score', nn.ReLU(inplace=True)),
                ('prob', nn.LogSoftmax(dim=1))
                ]))
            loadCheckpointDict(self.features_prob, reader) 

    def forward(self, x):
        x4_3 = self.features_conv4_3(x)
        x5_3 = self.features_conv5_3(x4_3)
        x5 = self.features_upscore_conv5(x5_3)
        x4 = self.features_score_conv4(x4_3)
        prob = self.features_prob(x4 + x5)
        return prob
        
