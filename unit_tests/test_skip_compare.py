# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:07:54 2018

@author: bokorn
"""
import numpy as np
import os
import datetime

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
log_dir = '/home/bokorn/results/test/'
current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(log_dir,current_timestamp)
    
model = gen_pose_net('alexnet', 'skip')
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
                    num_workers=4, 
                    batch_size=1, 
                    shuffle=True)

loader.dataset.loop_truth = [1,1]

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
num_epochs = 100
cum_batch_idx = 0

for epoch in range(num_epochs):
    for batch_idx, (images, trans, quats, models, model_files) in enumerate(loader):
    
        results01 = evaluatePairReg(model, images[0], images[1], trans[0],
                                    optimizer = optimizer, 
                                    disp_metrics = True)
        results10 = evaluatePairReg(model, images[1], images[0], trans[1],
                                    optimizer = optimizer, 
                                    disp_metrics = True)
    
        info = {}
        for k,v in results01.items():
            if('est' not in k):
                info[k] = v
        
        for k,v in results10.items():
            if('est' not in k):
                info[k] += v
                info[k] /= 2.0
        if(np.isnan(list(info.values())).any()):
            import IPython; IPython.embed()
        for tag, value in info.items():
            logger.scalar_summary(tag, value, cum_batch_idx+1)
            
#        origin_imgs = to_np(images[0].view(-1, 3, img_size[0], img_size[1])[:num_display_imgs])
#        query_imgs = to_np(images[1].view(-1, 3, img_size[0], img_size[1])[:num_display_imgs])
#        quat_true = to_np(trans[0][:num_display_imgs])
#        quat_est = to_np(results01['quat_est'][:num_display_imgs])
#        
#        render_imgs = renderQuaternions(origin_imgs, query_imgs, 
#                                        quat_true, quat_est,
#                                        origin_quats = quats[0][:num_display_imgs],
#                                       model_files=model_files[0][:num_display_imgs],
#                                        camera_dist = .75)
#                            
#                                              
#        disp_imgs = makeDisplayImages(origin_imgs, query_imgs,
#                                      None, quat_true,
#                                      None, quat_est, 
#                                      num_bins = (1,1,1))
#        
#        image_info = {'renders':render_imgs, 'display':disp_imgs}
#        
#        for tag, images in image_info.items():
#            logger.image_summary(tag, images, 1)
            cum_batch_idx += 1
import IPython; IPython.embed()