# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:07:54 2018

@author: bokorn
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.training.utils import evaluateLoopReg, evaluatePairReg, to_np
from generic_pose.models.pose_networks import gen_pose_net


import numpy as np
import cv2

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
                    batch_size=1, 
                    shuffle=True)

                                     
#loader.dataset.loop_truth = [1,0,0]
#j, (images, trans, quats, models, model_files) = next(enumerate(loader))
#loss = evaluateLoop(model, images, trans, loop_truth = [1,0,0])

loop_truth = [1,0,0]
loader.dataset.loop_truth = loop_truth
images, trans, quats, models, model_files = loader.dataset.__getitem__(0)
#j, (images, trans, quats, models, model_files) = next(enumerate(loader))

import IPython; IPython.embed()

cv2.imwrite("/home/bokorn/results/test/imgs/car1.png", np.transpose(to_np(images[0])*255, [1,2,0]))
cv2.imwrite("/home/bokorn/results/test/imgs/car2.png", np.transpose(to_np(images[1])*255, [1,2,0]))
cv2.imwrite("/home/bokorn/results/test/imgs/car3.png", np.transpose(to_np(images[2])*255, [1,2,0]))

import IPython; IPython.embed()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
loader.dataset.loop_truth = loop_truth
j, (images, trans, quats, models, model_files) = next(enumerate(loader))

loader.dataset.loop_truth = []
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
k, (origin, query, quat_true, class_true, origin_quat, model_file) = next(enumerate(loader))


loop_res = evaluateLoopReg(model, images, trans, loop_truth = [0,1], 
                            optimizer=None, disp_metrics=True)
print(loop_res['err_loop'])

pair_res = evaluatePairReg(model, origin, query, quat_true,
                           optimizer=None, disp_metrics=True)
                            
print(pair_res['err_quat'])

import IPython; IPython.embed()