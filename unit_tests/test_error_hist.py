# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:07:54 2018

@author: bokorn
"""
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.training.utils import evaluatePairReg
from generic_pose.models.pose_networks import gen_pose_net

from generic_pose.utils.display_pose import makeHistogramImages

data_folders = '/scratch/bokorn/data/renders/drill_1_renders/train'
model_data_file = '/scratch/bokorn/data/models/035_power_drill/google_64k/textured.obj'
weight_file = '/home/bokorn/results/test/weights/best_quat.pth'

model = gen_pose_net('alexnet', 'basic')
model.load_state_dict(torch.load(weight_file))

loader = DataLoader(PoseImageDataSet(data_folders=data_folders,
                                     img_size = (224, 224),
                                     model_filenames=model_data_file,
                                     background_filenames = None,
                                     classification = False,
                                     num_bins= (1,1,1),
                                     distance_sigma=1),
                    num_workers=4, 
                    batch_size=32, 
                    shuffle=True)

model.eval()
model.cuda()

errors = []
angles = []
cumulative_batch_idx = 0
while(True):
    for k, (origin, query, quat_true, class_true, origin_quat, model_file) in enumerate(loader):
        results = evaluatePairReg(model, origin, query, quat_true,
                                  optimizer = None, 
                                  disp_metrics = True)
    
        
        errors.append(results['errs_vec'])
        angles.append(results['diff_vec'])
        cumulative_batch_idx += 1
        

        if(cumulative_batch_idx > 100):
            print(cumulative_batch_idx)
            mean_hist, error_hist, count_hist = makeHistogramImages(np.concatenate(errors), np.concatenate(angles))
            cv2.imwrite("/home/bokorn/results/test/imgs/hist_mean.png", np.transpose(mean_hist[0], [1,2,0]))
            cv2.imwrite("/home/bokorn/results/test/imgs/hist_error.png", np.transpose(error_hist[0], [1,2,0]))
            cv2.imwrite("/home/bokorn/results/test/imgs/hist_count.png", np.transpose(count_hist[0], [1,2,0]))

        if(cumulative_batch_idx > 1000):
            break
        
mean_hist, error_hist, count_hist = makeHistogramImages(np.concatenate(errors), np.concatenate(angles))
cv2.imwrite("/home/bokorn/results/test/imgs/hist_mean.png", np.transpose(mean_hist[0], [1,2,0]))
cv2.imwrite("/home/bokorn/results/test/imgs/hist_error.png", np.transpose(error_hist[0], [1,2,0]))
cv2.imwrite("/home/bokorn/results/test/imgs/hist_count.png", np.transpose(count_hist[0], [1,2,0]))
import IPython; IPython.embed()    
    