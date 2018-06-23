# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:07:54 2018

@author: bokorn
"""

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam

from generic_pose.datasets.renderer_dataset import PoseRendererDataSet
from generic_pose.losses.quaternion_loss import loopConsistencyLoss, quaternionMultiply
from generic_pose.training.utils import evaluateLoopReg, evaluatePairReg
from generic_pose.models.pose_networks import gen_pose_net

from generic_pose.training.logger import Logger

import os
import time
import datetime

import numpy as np
import cv2

model_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/model_sets/cars_3_train.txt'
with open(model_file, 'r') as f:    
    model_folders = f.read().split()

results_dir = '/home/bokorn/results/test/'
current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(results_dir,current_timestamp,'log')
logger = Logger(log_dir)


model = gen_pose_net('alexnet', 'basic')
model.eval()
model.cuda()
optimizer = Adam(model.parameters(), lr=0.00001)

dataset = PoseRendererDataSet(model_folders=model_folders,
                              img_size = (224, 224),
                              background_filenames = None,
                              classification = False)
                                        
loader = DataLoader(dataset,
                    num_workers=32, 
                    batch_size=32, 
                    shuffle=True)

                                     
#loader.dataset.loop_truth = [1,0,0]
#j, (images, trans, quats, models, model_files) = next(enumerate(loader))
#loss = evaluateLoop(model, images, trans, loop_truth = [1,0,0])

loop_truth = [1,0,0]
loader.dataset.loop_truth = loop_truth
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
j, (images, trans, quats, models, model_files) = next(enumerate(loader))

loader.dataset.loop_truth = []
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
t = time.time()
k, (origin, query, quat_true, class_true, origin_quat, model_file) = next(enumerate(loader))
print('Load time: ', time.time()-t)
#t = time.time()
#loop_res = evaluateLoopReg(model, images, trans, loop_truth = [0,0,0], 
#                           optimizer=None, disp_metrics=True)
#print('Eval time: ', time.time()-t)

cv2.imwrite("/home/bokorn/results/test/imgs/car1.png", np.transpose(images[0], [1,2,0]))
cv2.imwrite("/home/bokorn/results/test/imgs/car2.png", np.transpose(images[1], [1,2,0]))
cv2.imwrite("/home/bokorn/results/test/imgs/car3.png", np.transpose(images[2], [1,2,0]))

import IPython; IPython.embed()

#image_info = {'img'+str(j):img for j, img in enumerate(images)}
#for tag, img in image_info.items():
for j, img in enumerate(images):
    logger.image_summary('loop', img, j)

import IPython; IPython.embed()

loop_res = evaluateLoopReg(model, images, trans, loop_truth = [0,1], 
                           optimizer=None, disp_metrics=True)
print(loop_res['err_loop'])

pair_res = evaluatePairReg(model, origin, query, quat_true,
                           optimizer=None, disp_metrics=True)
                            
print(pair_res['err_quat'])

import IPython; IPython.embed()