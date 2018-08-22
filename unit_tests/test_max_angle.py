# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:07:54 2018

@author: bokorn
"""
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.datasets.image_pair_dataset import PoseImagePairsDataSet

from generic_pose.losses.quaternion_loss import quaternionLoss
from generic_pose.training.utils import to_var
from generic_pose.models.pose_networks import gen_pose_net
from quat_math import quatAngularDiff

import generic_pose.utils.transformations as q_utils


data_folders = '/scratch/bokorn/data/renders/drill_1_renders/valid/'
model_data_file = '/scratch/bokorn/data/models/035_power_drill/google_64k/textured.obj'

model = gen_pose_net('alexnet', 'basic')
model.eval()
model.cuda()
optimizer = Adam(model.parameters(), lr=0.00001)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
dataset = PoseImageDataSet(data_folders=data_folders,
                           img_size = (224, 224),
                           model_filenames=model_data_file,
                           background_filenames = None,
                           classification = False,
                           num_bins= (1,1,1),
                           distance_sigma=1)

max_angle = 45.0/180.0*np.pi
iters = np.zeros(len(dataset))

#for model, indices in dataset.model_idxs.items():
#    quats = [np.load(dataset.data_filenames[idx] + '.npy') for idx in indices]


for index in range(len(dataset)):
    model = dataset.data_models[index]
    model_list_idx = dataset.data_model_list_idx[index]
    origin_quat = np.load(dataset.data_filenames[index] + '.npy')

    while(True):
        query_idx = dataset.model_idxs[model][model_list_idx - np.random.randint(0, len(dataset.model_idxs[model])-1)]
        query_quat = np.load(dataset.data_filenames[query_idx] + '.npy')
        iters[index] += 1
        if(quatAngularDiff(query_quat, origin_quat) < max_angle or iters[index] >= 200):
            break

import IPython; IPython.embed()
#origin = to_var(origin)
#query = to_var(query)
#quat_true = to_var(quat_true)
#
#optimizer.zero_grad()
#
#origin_features = model.features(origin)
#query_features = model.features(query)
#quat_est = model.compare_network(origin_features, 
#                                 query_features)
#
#loss_quat = quaternionLoss(quat_est, quat_true)

