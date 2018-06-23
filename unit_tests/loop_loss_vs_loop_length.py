# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 01:58:25 2018

@author: bokorn
"""

import numpy as np

import torch
from torch.utils.data import DataLoader

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.training.utils import evaluateLoopReg, evaluatePairReg
from generic_pose.models.pose_networks import gen_pose_net
import generic_pose.utils.transformations as tf_trans
import generic_pose.utils.transformations as tf_trans
from generic_pose.utils.data_preprocessing import quatAngularDiff, quat2AxisAngle

#weight_file = '/home/bokorn/results/alexnet_reg_loop_car3/2018-03-16_11-51-19/weights/best_quat.pth'
weight_file = '/home/bokorn/results/alexnet_reg_car3/2018-03-16_11-51-49/weights/best_quat.pth'
train_data_folder = '/scratch/bokorn/data/renders/car_3_renders/train'
model_data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/model_sets/cars_100_train.txt'
model_filenames = {}

with open(model_data_file, 'r') as f:    
    filenames = f.read().split()
for path in filenames:
    model = path.split('/')[-2]
    model_filenames[model] = path


model = gen_pose_net('alexnet', 'basic', output_dim = 4, pretrained = False)
model.load_state_dict(torch.load(weight_file))
model.eval()
model.cuda()

loader = DataLoader(PoseImageDataSet(data_folders=train_data_folder,
                                     img_size = (224, 224),
                                     model_filenames=model_filenames,
                                     background_filenames = None,
                                     classification = False,
                                     num_bins= (1,1,1),
                                     distance_sigma=1),
                    num_workers=4, 
                    batch_size=8, 
                    shuffle=True)


max_len = 50
num_samples = 8
loop_errors = np.zeros([max_len-1,num_samples])

for loop_length in range(2,max_len+1):
    if(loop_length > 2):
        loop_truth =  [1 for _ in range(loop_length-2)] + [0,0]
    else:
        loop_truth = [1,0]
    
    #loop_test = [1,] + [0 for _ in range(loop_length-1)]
    loop_test = np.zeros(loop_length)
    print(loop_truth, loop_test)
    loader.dataset.loop_truth = loop_truth
    for j, (images, trans, quats, models, model_files) in enumerate(loader):
        loop_res = evaluateLoopReg(model, images, trans, loop_truth = loop_test , optimizer=None, disp_metrics=True)    
        loop_errors[loop_length-2, j] = loop_res['err_loop']
        if(j == num_samples-1):
            break

import IPython; IPython.embed()
