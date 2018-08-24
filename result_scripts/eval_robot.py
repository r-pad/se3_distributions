# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:24:38 2018

@author: bokorn
"""
import torch
from torch.utils.data import DataLoader

import os

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.models.pose_networks import gen_pose_net
from generic_pose.training.utils import to_np, evaluatePairReg, evaluateLoopReg

import numpy as np
import cv2
import glob
import os

def getMaxCheckpoint(weights_folder):
    max_checkpoint = -float('inf')
    files = glob.glob(os.path.join(weights_folder,'checkpoint_*.pth'), recursive=True)
    for filename in files:
        cp = int(filename.split('.')[-2].split('_')[-1])
        max_checkpoint = max(max_checkpoint, cp)
    return max_checkpoint

#train_folder = '/ssd0/bokorn/data/renders/all_car_renders/train/'
#valid_folder = '/ssd0/bokorn/data/renders/all_car_renders/valid/'
#vclass_folder = '/ssd0/bokorn/data/renders/all_car_renders/valid/'

train_folder = '/ssd0/bokorn/data/robot_data/train'
valid_class_folder = '/ssd0/bokorn/data/robot_data/valid_model' 
valid_folder = '/ssd0/bokorn/data/robot_data/valid_pose'

model_filenames = {}

max_orientation_offset = None;

train_loader = DataLoader(PoseImageDataSet(data_folders=train_folder,
                                           img_size = (224,224),
                                           max_orientation_offset = max_orientation_offset,
                                           max_orientation_iters = 200,
                                           model_filenames=None,
                                           background_filenames = None,
                                           classification=False,
                                           num_bins=(1,1,1),
                                           distance_sigma=1),
                                       num_workers=4, 
                                       batch_size=32, 
                                       shuffle=True)
train_loader.dataset.loop_truth = [1,0,0]
        
valid_loader = DataLoader(PoseImageDataSet(data_folders=valid_folder,
                                           img_size = (224,224),
                                           max_orientation_offset = max_orientation_offset,
                                           max_orientation_iters = 200,
                                           model_filenames=None,
                                           background_filenames = None,
                                           classification=False,
                                           num_bins=(1,1,1),
                                           distance_sigma=1),
                           num_workers=4, 
                           batch_size=32, 
                           shuffle=True)
valid_loader.dataset.loop_truth = [1,0,0]

vclass_loader = DataLoader(PoseImageDataSet(data_folders=valid_class_folder,
                                           img_size = (224,224),
                                           max_orientation_offset = max_orientation_offset,
                                           max_orientation_iters = 200,
                                           model_filenames=None,
                                           background_filenames = None,
                                           classification=False,
                                           num_bins=(1,1,1),
                                           distance_sigma=1),
                           num_workers=4, 
                           batch_size=32, 
                           shuffle=True)
vclass_loader.dataset.loop_truth = [1,0,0]

results_folder = '/home/bokorn/results/result_imgs/eval_robot/' 

weight_paths = ['/home/bokorn/results/robot/alex_reg_robot_plane_maxdot_45deg_quat/2018-04-25_11-36-29/weights',
                '/home/bokorn/results/robot/alex_reg_robot_plane_45deg/2018-04-25_11-30-03/weights',
                ]


weight_filenames = []
experiment_name = []
for path in weight_paths:
    checkpoint = getMaxCheckpoint(path)
    weight_filenames.append(os.path.join(path, 'checkpoint_{}.pth'.format(checkpoint)))
    experiment_name.append(path.split('/')[-3])

#import IPython; IPython.embed()
num_iter = 200
for name, weight_file in zip(experiment_name, weight_filenames):
    print(name)
    model = gen_pose_net('alexnet', 'basic', output_dim = 4)
    model.load_state_dict(torch.load(weight_file))
    model.eval()
    model.cuda()
    
    train_errs = []
    for batch_idx, (images, trans, quats, models, model_files) in enumerate(train_loader):
        train_results = evaluatePairReg(model, images[0], images[1], trans[0],
                                        optimizer = None, disp_metrics = True)
        train_errs.append(train_results['mean_err'])
        if(batch_idx == num_iter-1):
            break

    valid_errs = []

    for batch_idx, (images, trans, quats, models, model_files) in enumerate(valid_loader):
        valid_results = evaluatePairReg(model, images[0], images[1], trans[0],
                                        optimizer = None, disp_metrics = True)
        valid_errs.append(valid_results['mean_err'])
        if(batch_idx == num_iter-1):
            break

    vclass_errs = []

    for batch_idx, (images, trans, quats, models, model_files) in enumerate(vclass_loader):
        vclass_results = evaluatePairReg(model, images[0], images[1], trans[0],
                                        optimizer = None, disp_metrics = True)
        vclass_errs.append(vclass_results['mean_err'])
        if(batch_idx == num_iter-1):
            break
        
    filename = os.path.join(results_folder, name+'_errs.npz')
    np.savez(filename, 
             train_errs=train_errs,
             valid_errs=valid_errs,
             vclass_errs=vclass_errs)
    #filename = os.path.join(results_folder, name+'_imgs.npz')
    #np.savez(filename, 
    #         train_imgs=train_imgs,
    #         train_tgts=train_tgts,
    #         valid_imgs=valid_imgs,
    #         valid_tgts=valid_tgts)

train_loader.dataset.max_orientation_offset = np.pi/4.0
valid_loader.dataset.max_orientation_offset = np.pi/4.0
vclass_loader.dataset.max_orientation_offset = np.pi/4.0

for name, weight_file in zip(experiment_name, weight_filenames):
    print(name)
    model = gen_pose_net('alexnet', 'basic', output_dim = 4)
    model.load_state_dict(torch.load(weight_file))
    model.eval()
    model.cuda()
    
    train_errs = []
    for batch_idx, (images, trans, quats, models, model_files) in enumerate(train_loader):
        train_results = evaluatePairReg(model, images[0], images[1], trans[0],
                                        optimizer = None, disp_metrics = True)
        train_errs.append(train_results['mean_err'])
        if(batch_idx == num_iter-1):
            break

    valid_errs = []

    for batch_idx, (images, trans, quats, models, model_files) in enumerate(valid_loader):
        valid_results = evaluatePairReg(model, images[0], images[1], trans[0],
                                        optimizer = None, disp_metrics = True)
        valid_errs.append(valid_results['mean_err'])
        if(batch_idx == num_iter-1):
            break

    vclass_errs = []

    for batch_idx, (images, trans, quats, models, model_files) in enumerate(vclass_loader):
        vclass_results = evaluatePairReg(model, images[0], images[1], trans[0],
                                        optimizer = None, disp_metrics = True)
        vclass_errs.append(vclass_results['mean_err'])
        if(batch_idx == num_iter-1):
            break
        
    filename = os.path.join(results_folder, name+'45_errs.npz')
    np.savez(filename, 
             train_errs=train_errs,
             valid_errs=valid_errs,
             vclass_errs=vclass_errs)
import IPython; IPython.embed()

base_all = np.load(results_folder + 'alex_reg_robot_plane_maxdot_45deg_quat_errs.npz')
base_45 = np.load(results_folder + 'alex_reg_robot_plane_maxdot_45deg_quat_45_errs.npz')

small_all = np.load(results_folder + 'alex_reg_robot_plane_45deg_errs.npz')
small_45 = np.load(results_folder + 'alex_reg_robot_plane_45deg_45_errs.npz')

