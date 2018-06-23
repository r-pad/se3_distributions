# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:37:13 2018

@author: bokorn
"""

import numpy as np

import torch
import torch.nn as nn

features_regression_size = 512

from generic_pose.models.symetric_layers import QSymetric, SplitLinear
from generic_pose.models.compare_networks import CompareNet, SymetricCompareNet, SplitCompareNet
from generic_pose.models.pose_networks import gen_pose_net

from torch.autograd import Variable
from torch.utils.data import DataLoader
from generic_pose.datasets.image_dataset import PoseImageDataSet

compare_net = nn.Sequential(#nn.Dropout(),
                            nn.Linear(features_regression_size * 2, 4096),
                            nn.ReLU(inplace=True),
                            #nn.Dropout(),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True),
                            nn.Linear(4096, 4))

compare_tanh = nn.Sequential(#nn.Dropout(),
                             SplitLinear(512, 256, 1024, 512),
                             nn.Tanh(),
                             #nn.Dropout(),
                             SplitLinear(1024, 512, 1024, 512),
                             nn.Tanh(),
                             SplitLinear(1024, 512, 3, 1))


q_symetric = QSymetric(features_regression_size, 512, 256)

data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/robot_plane_valid.txt'
with open(data_file, 'r') as f:    
    data_folders = f.read().split()
model_data_file = '/scratch/bokorn/data/models/sawyer_gripper/electric_gripper_w_fingers.obj'

loader = DataLoader(PoseImageDataSet(data_folders=data_folders,
                                     img_size = (224, 224),
                                     model_filenames=model_data_file,
                                     background_filenames = None,
                                     classification = False,
                                     num_bins= (1,1,1),
                                     distance_sigma=1),
                    num_workers=4, 
                    batch_size=10, 
                    shuffle=True)

np.random.seed(0)
neg_err = 0
sym_err = 0
values = []
split = 3
#split = 512

#comp_net = CompareNet(features_regression_size, output_dim = 4)
comp_net = gen_pose_net('alexnet', 'basic', output_dim = 4)
comp_net.eval()
#sym_net = SymetricCompareNet(features_regression_size, output_dim = 4)
sym_net = gen_pose_net('alexnet', 'symetric', output_dim = 4)
sym_net.eval()
#split_net = SplitCompareNet(features_regression_size, output_dim = 4)
split_net = gen_pose_net('alexnet', 'split', output_dim = 4)
split_net.eval()

model = split_net

#for j in range(100):
#    v1 = torch.autograd.Variable(torch.Tensor(np.random.randn(100,features_regression_size)))
#    v2 = torch.autograd.Variable(torch.Tensor(np.random.randn(100,features_regression_size)))
#    
for j, (origin, query, quat_true, class_true, origin_quat, model_file) in enumerate(loader):
    #q_symetric = QSymetric(features_regression_size, 512, 256)
    origin = Variable(origin)
    query = Variable(query)
    cs12 = model(origin, query)
    cs21 = model(query, origin)
    #cs12 = compare_tanh(q_symetric(v1, v2))
    #cs21 = compare_tanh(q_symetric(v2, v1))
#    cs12 = q_symetric(v1, v2)
#    cs21 = q_symetric(v2, v1)
    values += [(cs12.data.numpy(), cs21.data.numpy())]
    
    neg_err += torch.abs(cs12[:,:split]+cs21[:,:split]).sum().data.numpy().sum()
    sym_err += torch.abs(cs12[:,split:]-cs21[:,split:]).sum().data.numpy().sum()
    
    if(torch.abs(cs12[:,:split]+cs21[:,:split]).sum().data.numpy() > 1e-9):
        print('Error Negated {}'.format(torch.abs(cs12[:,:split]+cs21[:,:split]).sum()))
        print(cs12[0])
        print(cs21[0])
    if(torch.abs(cs12[:,split:]-cs21[:,split:]).sum().data.numpy() > 1e-9):
        print('Error Symetric {}'.format(torch.abs(cs12[:,split:]-cs21[:,split:]).sum()))
        print(cs12[0])
        print(cs21[0])
    if(j > 100):
        break
print('Neg Error:', neg_err)
print('Sym Error:', sym_err)
import IPython; IPython.embed()
