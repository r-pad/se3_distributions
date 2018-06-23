# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 21:50:13 2018

@author: bokorn
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.training.max_angle_utils import evaluateMaxedDotAxisAngle, evaluateMaxedMultAxisAngle, evaluateMaxedTrueAxisAngle
from generic_pose.models.pose_networks import gen_pose_net


import numpy as np
import cv2

from generic_pose.losses.quaternion_loss import quaternionInverse, quaternionMultiply, clipQuatAngle, axisAngle2Quat
from generic_pose.training.utils import to_var, to_np
from generic_pose.utils.data_preprocessing import quatDiff, quat2AxisAngle
from generic_pose.utils.transformations import quaternion_multiply, quaternion_inverse

data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/folder_sets/car_3_models_train.txt'
with open(data_file, 'r') as f:    
    data_folders = f.read().split()

model_data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/model_sets/cars_100_train.txt'
model_filenames = {}

with open(model_data_file, 'r') as f:    
    filenames = f.read().split()
for path in filenames:
    model = path.split('/')[-2]
    model_filenames[model] = path
    
model = gen_pose_net('alexnet', 'basic')
model.eval()
model.cuda()
optimizer = Adam(model.parameters(), lr=0.00001)

loader = DataLoader(PoseImageDataSet(data_folders=data_folders,
                                     img_size = (224, 224),
                                     model_filenames=model_filenames,
                                     background_filenames = None,
                                     classification = False,
                                     num_bins= (1,1,1),
                                     distance_sigma=1),
                    num_workers=4, 
                    batch_size=4, 
                    shuffle=True)

loop_length = 3
loader.dataset.loop_truth = [1,] + [0 for _ in range(loop_length-1)]

for j, (images, trans, quats, models, model_files) in enumerate(loader):

    dot_res = evaluateMaxedDotAxisAngle(model, images[0], images[1], trans[0],
                               optimizer=None, disp_metrics=True)
                                
    mult_res = evaluateMaxedMultAxisAngle(model, images[0], images[1], 
                                 quats[0], quats[1],
                                 optimizer=None, disp_metrics=True)
                                
    true_res = evaluateMaxedTrueAxisAngle(model, images[0], images[1], trans[0],
                                 optimizer=None, disp_metrics=True)
    print(dot_res['mean_err'], mult_res['mean_err'], true_res['mean_err'])
    print(dot_res['loss_quat'], mult_res['loss_quat'], true_res['loss_quat'])
    if(j >= 10):
        break
import IPython; IPython.embed()