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
from generic_pose.utils.data_preprocessing import quatDiff

import generic_pose.utils.transformations as q_utils
import generic_pose.utils.data_preprocessing as pproc

data_folders = '/scratch/bokorn/data/renders/drill_1_renders/valid/'

model_data_file = '/scratch/bokorn/data/models/035_power_drill/google_64k/textured.obj'

model = gen_pose_net('alexnet', 'basic')
model.eval()
model.cuda()
optimizer = Adam(model.parameters(), lr=0.00001)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
loader = DataLoader(PoseImageDataSet(data_folders=data_folders,
                                     img_size = (224, 224),
                                     model_filenames=model_data_file,
                                     background_filenames = None,
                                     classification = False,
                                     num_bins= (1,1,1),
                                     distance_sigma=1),
                    num_workers=4, 
                    batch_size=1, 
                    shuffle=True)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
pair_loader = DataLoader(PoseImagePairsDataSet(data_folders=data_folders,
                                               img_size = (224, 224),
                                               model_filenames=model_data_file,
                                               background_filenames = None,
                                               classification = False,
                                               num_bins= (1,1,1),
                                               distance_sigma=1),
                         num_workers=4, 
                         batch_size=1, 
                         shuffle=True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
k1, (origin1, query1, quat_true1, class_true1, origin_quat1, model_file1) = next(enumerate(loader))
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

for k, (origin, query, quat_true, class_true, origin_quat, model_file, query_quat) in enumerate(pair_loader):
    q1 = quat_true.numpy()[0]
    q1 *= np.sign(q1[3])
    q2 = quatDiff(query_quat.numpy()[0], origin_quat.numpy()[0])
    
    diff = np.sum(np.abs(q1 - q2))
    if(diff > 1e-9):
        print('ERROR: {}'.format(diff))

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

import IPython; IPython.embed()