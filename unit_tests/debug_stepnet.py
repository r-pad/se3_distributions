# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:41:51 2018

@author: bokorn
"""

import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, Adadelta, SGD

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.losses.quaternion_loss import quaternionLoss
from generic_pose.training.utils import to_var
from generic_pose.models.pose_networks import gen_pose_net
from generic_pose.utils.data_preprocessing import quatAngularDiff
from generic_pose.training.step_utils import getAxes, evaluateStepClass

import generic_pose.utils.transformations as q_utils
import generic_pose.utils.data_preprocessing as pproc

train_data_folder = '/ssd0/bokorn/data/renders/drill_debug/train' 
valid_pose_folder = '/ssd0/bokorn/data/renders/drill_debug/valid'
model_data_file = '/scratch/bokorn/data/models/035_power_drill/google_64k/textured.obj'

dataset = PoseImageDataSet(data_folders=train_data_folder,
                           img_size = (224, 224),
                           model_filenames=model_data_file,
                           background_filenames = None,
                           classification = False,
                           num_bins= (1,1,1),
                           distance_sigma=1)
loader = DataLoader(dataset, num_workers=1, batch_size=1, shuffle=False)
loader.dataset.loop_truth = [1,1]

model = gen_pose_net('alexnet', 'basic', output_dim = 7, pretrained = False)
model.train()
model.cuda()

optimizer = Adam(model.parameters(), lr=0.00001)


step_angle = np.pi/4
bin_axes = getAxes(6)
for j in range(1000):
    for batch_idx, (images, trans, quats, models, model_files) in enumerate(loader):
        train_results = evaluateStepClass(model,
                                          images[0], images[1], 
                                          quats[0], quats[1],
                                          bin_axes = bin_axes,
                                          step_angle = step_angle,
                                          use_softmax_labels=False,
                                          loss_type='mse',
                                          optimizer = optimizer, 
                                          disp_metrics = False)

import IPython; IPython.embed()
