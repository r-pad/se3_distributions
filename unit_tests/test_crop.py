# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:07:54 2018

@author: bokorn
"""
import numpy as np
import cv2

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam

from generic_pose.datasets.image_dataset import PoseImageDataSet

from generic_pose.losses.quaternion_loss import quaternionLoss
from generic_pose.training.utils import to_var, to_np
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
                           crop_percent = .85,
                           model_filenames=model_data_file,
                           background_filenames = None,
                           classification = False,
                           num_bins= (1,1,1),
                           distance_sigma=1)

max_angle = 45.0/180.0*np.pi
iters = np.zeros(len(dataset))

origin_img, query_img, offset_quat, offset_class, origin_quat, model_file = dataset.getPair(0)
cv2.imwrite('/home/bokorn/results/test/imgs/o_0.png' , 255*to_np(np.transpose(origin_img,[1,2,0])))
cv2.imwrite('/home/bokorn/results/test/imgs/q_0.png' , 255*to_np(np.transpose(query_img,[1,2,0])))

origin_img, query_img, offset_quat, offset_class, origin_quat, model_file = dataset.getPair(1)
cv2.imwrite('/home/bokorn/results/test/imgs/o_1.png' , 255*to_np(np.transpose(origin_img,[1,2,0])))
cv2.imwrite('/home/bokorn/results/test/imgs/q_1.png' , 255*to_np(np.transpose(query_img,[1,2,0])))

origin_img, query_img, offset_quat, offset_class, origin_quat, model_file = dataset.getPair(2)
cv2.imwrite('/home/bokorn/results/test/imgs/o_2.png' , 255*to_np(np.transpose(origin_img,[1,2,0])))
cv2.imwrite('/home/bokorn/results/test/imgs/q_2.png' , 255*to_np(np.transpose(query_img,[1,2,0])))
import IPython; IPython.embed()
