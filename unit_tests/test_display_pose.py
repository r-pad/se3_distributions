# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:07:54 2018

@author: bokorn
"""

import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.training.utils import to_var, to_np, evaluatePairReg
from generic_pose.models.pose_networks import gen_pose_net
from generic_pose.training.logger import Logger
from generic_pose.utils.display_pose import makeDisplayImages, renderQuaternions


data_folders = '/scratch/bokorn/data/renders/drill_1_renders/valid/'
model_data_file = '/scratch/bokorn/data/models/035_power_drill/google_64k/textured.obj'
#model_data_file = '/scratch/bokorn/data/models/sawyer_gripper/electric_gripper_w_fingers.obj'
log_dir = '/home/bokorn/results/test/log/'

model = gen_pose_net('alexnet', 'basic')
model.eval()
model.cuda()
optimizer = Adam(model.parameters(), lr=0.00001)

logger = Logger(log_dir)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
img_size = (224, 224)
num_display_imgs = 1

loader = DataLoader(PoseImageDataSet(data_folders=data_folders,
                                     img_size = img_size,
                                     model_filenames=model_data_file,
                                     background_filenames = None,
                                     classification = False,
                                     num_bins= (1,1,1),
                                     distance_sigma=1),
                    num_workers=32, 
                    batch_size=32, 
                    shuffle=True)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
k1, (origin, query, quat_true, class_true, origin_quat, model_file) = next(enumerate(loader))
results = evaluatePairReg(model, origin, query, quat_true,
                          optimizer = optimizer, 
                          disp_metrics = True)


origin_imgs = to_np(origin.view(-1, 3, img_size[0], img_size[1])[:num_display_imgs])
query_imgs = to_np(query.view(-1, 3, img_size[0], img_size[1])[:num_display_imgs])
quat_true = to_np(quat_true[:num_display_imgs])
quat_est = to_np(results['quat_est'][:num_display_imgs])

render_imgs = renderQuaternions(origin_imgs, query_imgs, 
                                quat_true, quat_true,
                                origin_quats = origin_quat[:num_display_imgs],
                                model_files=model_file[:num_display_imgs],
                                camera_dist = .75)
                    
                                      
disp_imgs = makeDisplayImages(origin_imgs, query_imgs,
                              None, quat_true,
                              None, quat_est, 
                              num_bins = (1,1,1))

image_info = {'renders':render_imgs, 'display':disp_imgs}

for tag, images in image_info.items():
    logger.image_summary(tag, images, 1)
    
cv2.imwrite("/home/bokorn/results/test/imgs/disp_render.png", np.transpose(render_imgs[0], [1,2,0])*255)

import IPython; IPython.embed()