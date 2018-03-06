# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:07:35 2018

@author: bokorn
"""

#Skip connections
import torch.nn as nn

def create_combine_net(inplanes, planes, blocks):
   class BasicBlock(nn.Module):
     def __init__(self, inplanes, planes, downsample=None):
       super(BasicBlock, self).__init__()
       self.lin1 = nn.Linear(inplanes, planes)
       self.bn1 = nn.BatchNorm1d(planes)
       self.lin2 = nn.Linear(planes, planes)
       self.bn2 = nn.BatchNorm1d(planes)
       self.relu = nn.ReLU(inplace=True)
       self.downsample = downsample
     
     def forward(self, x):
       residual = x

       out = self.lin1(x)
       out = self.bn1(out)
       out = self.relu(out)

       out = self.lin2(out)
       out = self.bn2(out)

       if self.downsample is not None:
         residual = self.downsample(x)

       out += residual
       out = self.relu(out)
       
       return out
   
   def _make_layer(inplanes, planes, blocks):
     downsample = None
     if inplanes != planes:
       downsample = nn.Sequential(
           nn.Linear(inplanes, planes),
           nn.BatchNorm1d(planes))
     
     layers = []
     layers.append(BasicBlock(inplanes, planes, downsample))
     for i in range(1, blocks):
       layers.append(BasicBlock(planes, planes))
     
     return nn.Sequential(*layers)
 
   # Main
   assert(len(planes) == len(blocks))
   layers = []
   for i in range(len(planes)):
     layers.append(_make_layer(inplanes, planes[i], blocks[i]))
     inplanes = planes[i]
   return nn.Sequential(*layers)
   
   

#combine_net = create_combine_net(
#       2 * NUM_CHANNELS,
#       planes=[1536, 512],
#       blocks=[1, 1])