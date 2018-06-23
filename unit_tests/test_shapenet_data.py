# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:39:27 2018

@author: bokorn
"""
import numpy as np
import os, glob, shutil
from torch.utils.data import DataLoader
from generic_pose.datasets.image_dataset import PoseImageDataSet

#data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/folder_sets/car_3_models_valid.txt'
#with open(data_file, 'r') as f:    
#    data_folders = f.read().split()
#files = []
#for folder in data_folders:
#    files.extend(glob.glob(folder + '/**/*.png', recursive=True))

#data_folder = '/scratch/bokorn/data/renders/car_42_renders/train'
data_folder = '/scratch/bokorn/data/renders/all_classes/train'
files = glob.glob(data_folder + '/**/*.png', recursive=True)

model_hierarchy = {}

for path in files:
    [model_class, model, filename] = '.'.join(path.split('.')[:-1]).split('/')[-3:]

    if(model_class not in model_hierarchy):
        model_hierarchy[model_class] = {model:[]}

    if(model not in model_hierarchy[model_class]):
        model_hierarchy[model_class][model] = []
        
    model_hierarchy[model_class][model].append('.'.join(path.split('.')[:-1]))

model_data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/model_sets/all_class_train.txt'
img_size = (224, 224)
img_loader = DataLoader(PoseImageDataSet(data_folders=[data_folder],
                                         img_size = img_size,
                                         model_filenames=model_data_file,
                                         background_filenames = None,
                                         classification = False,
                                         num_bins= (1,1,1),
                                         distance_sigma=1),
                         num_workers=32, 
                         batch_size=32, 
                         shuffle=True)

import IPython; IPython.embed()

#pose_valid = {}
#for cls in model_hierarchy.keys():
#    pose_valid[cls] = {}
#    for mdl in model_hierarchy[cls].keys():
#        num_images = len(model_hierarchy[cls][mdl])
#        pose_valid[cls][mdl] = model_hierarchy[cls][mdl][-int(num_images/10):]
#        del model_hierarchy[cls][mdl][-int(num_images/10):]
#
#import IPython; IPython.embed()
#
##train_dir = '/scratch/bokorn/data/renders/car_3_renders/valid'
##if not os.path.exists(train_dir):
##    os.makedirs(train_dir)
##    
##for cls in model_hierarchy.keys():
##    cls_path = os.path.join(train_dir,cls)
##    if not os.path.exists(cls_path):
##        os.makedirs(cls_path)
##    for mdl in model_hierarchy[cls].keys():
##        mdl_path = os.path.join(train_dir,cls,mdl)
##        if not os.path.exists(mdl_path):
##            os.makedirs(mdl_path)
##        for path in model_hierarchy[cls][mdl]:
##            filename = path.split('/')[-1]
##            shutil.copy(path+'.png', os.path.join(mdl_path, filename+'.png'))
##            shutil.copy(path+'.npy', os.path.join(mdl_path, filename+'.npy'))
#
#valid_pose_dir = '/scratch/bokorn/data/renders/car_42_renders/valid_pose'
#if not os.path.exists(valid_pose_dir):
#    os.makedirs(valid_pose_dir)
#
#for cls in pose_valid.keys():
#    cls_path = os.path.join(valid_pose_dir,cls)
#    if not os.path.exists(cls_path):
#        os.makedirs(cls_path)
#    for mdl in pose_valid[cls].keys():
#        mdl_path = os.path.join(valid_pose_dir,cls,mdl)
#        if not os.path.exists(mdl_path):
#            os.makedirs(mdl_path)
#        for path in pose_valid[cls][mdl]:
#            filename = path.split('/')[-1]
#            shutil.move(path+'.png', os.path.join(mdl_path, filename+'.png'))
#            shutil.move(path+'.npy', os.path.join(mdl_path, filename+'.npy'))