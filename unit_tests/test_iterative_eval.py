# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:24:38 2018

@author: bokorn
"""

import torch

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.models.pose_networks import gen_pose_net
from generic_pose.eval.iterative_evaluation import evaluateIterQuatModel

import numpy as np
import cv2
import glob

#train_data_folder = '/scratch/bokorn/data/renders/car_42_renders/train' 
#valid_model_folder = '/scratch/bokorn/data/renders/car_42_renders/valid'
#valid_pose_folder = '/scratch/bokorn/data/renders/car_42_renders/valid_pose'
#
#model_data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/model_sets/cars_100_train.txt'
#model_filenames = {}
#
#with open(model_data_file, 'r') as f:    
#    filenames = f.read().split()
#for path in filenames:
#    model = path.split('/')[-2]
#    model_filenames[model] = path
    
#train_files = glob.glob(train_data_folder + '/**/*.png', recursive=True)
#valid_model_files = glob.glob(valid_model_folder + '/**/*.png', recursive=True)
#valid_pose_files = glob.glob(valid_pose_folder + '/**/*.png', recursive=True)
#
#train_models = set()
#for path in train_files:
#    [model_class, model_name] = path.split('/')[-3:-1]
#    train_models.add(model_name)
#train_models = list(train_models)
#valid_model_models = set()
#for path in valid_model_files:
#    [model_class, model_name] = path.split('/')[-3:-1]
#    valid_model_models.add(model_name)
#valid_model_models = list(valid_model_models)
#valid_pose_models = set()
#for path in valid_pose_files:
#    [model_class, model_name] = path.split('/')[-3:-1]
#    valid_pose_models.add(model_name)
#valid_pose_models = list(valid_pose_models)

weight_file = '/home/bokorn/results/alex_reg_car42_maxdot_45deg_quat/2018-04-11_06-22-46/weights/checkpoint_312000.pth'
#train
model_file = '/home/bokorn/data/models/shapenetcore/02958343/1e17510abe8f98bea697d949e28d773c/model.obj'
##valid
#model_file = '/home/bokorn/data/models/shapenetcore/02958343/1a4ef4a2a639f172f13d1237e1429e9e/model.obj'
#model_file = model_filenames[train_models[0]]
[errs, imgs] = evaluateIterQuatModel(weight_file, model_file)
import IPython; IPython.embed()
