# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:07:54 2018

@author: bokorn
"""

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.losses.quaternion_loss import quaternionLoss
from generic_pose.training.utils import to_var
from generic_pose.models.pose_networks import gen_pose_net

import generic_pose.utils.transformations as q_utils
import generic_pose.utils.data_preprocessing as pproc

data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/robot_plane_valid.txt'
with open(data_file, 'r') as f:    
    data_folders = f.read().split()
model_data_file = '/scratch/bokorn/data/models/sawyer_gripper/electric_gripper_w_fingers.obj'

model = gen_pose_net('alexnet', 'basic')
model.eval()
model.cuda()
optimizer = Adam(model.parameters(), lr=0.00001)

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

k, (origin, query, quat_true, class_true, origin_quat, model_file) = next(enumerate(loader))

origin = to_var(origin)
query = to_var(query)
quat_true = to_var(quat_true)

optimizer.zero_grad()

origin_features = model.features(origin)
query_features = model.features(query)
quat_est = model.compare_network(origin_features, 
                                 query_features)

loss_quat = quaternionLoss(quat_est, quat_true)

import IPython; IPython.embed()