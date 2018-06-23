# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 01:48:41 2018

@author: bokorn
"""

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
from generic_pose.datasets.renderer_dataset import PoseRendererDataSet

from generic_pose.training.utils import to_var, to_np, evaluatePairReg
from generic_pose.models.pose_networks import gen_pose_net
from generic_pose.training.logger import Logger

import time


data_folders = '/scratch/bokorn/data/renders/drill_1_renders/valid/'
model_data_file = '/scratch/bokorn/data/models/035_power_drill/google_64k/textured.obj'
#model_data_file = '/scratch/bokorn/data/models/sawyer_gripper/electric_gripper_w_fingers.obj'
model_file = '/ssd0/bokorn/data/models/shapenetcore/02958343/'
#model_data_folders = ['/scratch/bokorn/data/models/sawyer_gripper/']

log_dir = '/home/bokorn/results/test/log/'

model = gen_pose_net('alexnet', 'basic')
model.eval()
model.cuda()
optimizer = Adam(model.parameters(), lr=0.00001)


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
img_size = (224, 224)
num_display_imgs = 1
num_iter = 5

img_loader = DataLoader(PoseImageDataSet(data_folders=data_folders,
                                     img_size = img_size,
                                     model_filenames=model_data_file,
                                     background_filenames = None,
                                     classification = False,
                                     num_bins= (1,1,1),
                                     distance_sigma=1),
                         num_workers=1, 
                         batch_size=16, 
                         shuffle=True)

                                        
render_loader = DataLoader(PoseRendererDataSet(model_folders=model_file,
                                               img_size = (224, 224),
                                               background_filenames = None,
                                               classification = False),
                           num_workers=1, 
                           batch_size=16, 
                           shuffle=True)
                    
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
img_load_times = []
img_eval_times = []
print('Image Dataset')
t = time.time()
for k, (origin, query, quat_true, class_true, origin_quat, model_file) in enumerate(img_loader):
    load_time = time.time()-t
    img_load_times.append(load_time)
    print('Load time: ', load_time)
    t = time.time()
    results = evaluatePairReg(model, origin, query, quat_true,
                              optimizer = optimizer, 
                              disp_metrics = False)
    
    eval_time = time.time()-t
    img_eval_times.append(eval_time)
    print('Eval time: ', eval_time)
    if(k > num_iter):
        break
    t = time.time()
    


print('Renderer Dataset')
render_load_times = []
render_eval_times = []
t = time.time()
for k, (origin_r, query_r, quat_true_r, class_true_r, origin_quat_r, model_file_r) in enumerate(render_loader):
    load_time = time.time()-t
    render_load_times.append(load_time)
    print('Load time: ', load_time)
    t = time.time()
    results = evaluatePairReg(model, origin_r, query_r, quat_true_r,
                              optimizer = optimizer, 
                              disp_metrics = False)
    eval_time = time.time()-t
    render_eval_times.append(eval_time)
    print('Eval time: ', eval_time)
    if(k > num_iter):
        break
    t = time.time()

import IPython; IPython.embed()
