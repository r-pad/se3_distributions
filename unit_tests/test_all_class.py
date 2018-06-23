# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:39:27 2018

@author: bokorn
"""
import numpy as np
import os, glob, shutil

#data_folder = '/scratch/bokorn/data/renders/all_classes'
#files = glob.glob(data_folder + '/**/*.png', recursive=True)
#npy_files = glob.glob(data_folder + '/**/*.npy', recursive=True)

files = np.load('files.npy')

model_hierarchy = {}

for path in files:
    [model_class, model, filename] = '.'.join(path.split('.')[:-1]).split('/')[-3:]

    if(model_class not in model_hierarchy):
        model_hierarchy[model_class] = {model:[]}

    if(model not in model_hierarchy[model_class]):
        model_hierarchy[model_class][model] = []
        
    model_hierarchy[model_class][model].append('.'.join(path.split('.')[:-1]))
        
class_valid = {'04468005':model_hierarchy['04468005']}
del model_hierarchy['04468005']

model_valid = {}
for cls in model_hierarchy.keys():
    num_models = len(model_hierarchy[cls])
    model_valid[cls] = {}
    v_mdls = []
    for j, mdl in enumerate(model_hierarchy[cls].keys()):
        if(j > num_models * 0.9):
            model_valid[cls][mdl] = model_hierarchy[cls][mdl]
            v_mdls.append(mdl)
    for mdl in v_mdls:
        del model_hierarchy[cls][mdl]
    
pose_valid = {}
for cls in model_hierarchy.keys():
    pose_valid[cls] = {}
    for mdl in model_hierarchy[cls].keys():
        num_images = len(model_hierarchy[cls][mdl])
        pose_valid[cls][mdl] = model_hierarchy[cls][mdl][-int(num_images/10):]
        del model_hierarchy[cls][mdl][-int(num_images/10):]

import IPython; IPython.embed()

valid_class_dir = '/scratch/bokorn/data/renders/all_classes/valid_class'
if not os.path.exists(valid_class_dir):
    os.makedirs(valid_class_dir)

for cls in class_valid.keys():
    cls_path = os.path.join(valid_class_dir,cls)
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)
    for mdl in class_valid[cls].keys():
        mdl_path = os.path.join(valid_class_dir,cls,mdl)
        if not os.path.exists(mdl_path):
            os.makedirs(mdl_path)
        for path in class_valid[cls][mdl]:
            filename = path.split('/')[-1]
            shutil.move(path+'.png', os.path.join(mdl_path, filename+'.png'))
            shutil.move(path+'.npy', os.path.join(mdl_path, filename+'.npy'))

valid_model_dir = '/scratch/bokorn/data/renders/all_classes/valid_model'
if not os.path.exists(valid_model_dir):
    os.makedirs(valid_model_dir)

for cls in model_valid.keys():
    cls_path = os.path.join(valid_model_dir,cls)
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)
    for mdl in model_valid[cls].keys():
        mdl_path = os.path.join(valid_model_dir,cls,mdl)
        if not os.path.exists(mdl_path):
            os.makedirs(mdl_path)
        for path in model_valid[cls][mdl]:
            filename = path.split('/')[-1]
            shutil.move(path+'.png', os.path.join(mdl_path, filename+'.png'))
            shutil.move(path+'.npy', os.path.join(mdl_path, filename+'.npy'))

valid_pose_dir = '/scratch/bokorn/data/renders/all_classes/valid_pose'
if not os.path.exists(valid_pose_dir):
    os.makedirs(valid_pose_dir)

for cls in pose_valid.keys():
    cls_path = os.path.join(valid_pose_dir,cls)
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)
    for mdl in pose_valid[cls].keys():
        mdl_path = os.path.join(valid_pose_dir,cls,mdl)
        if not os.path.exists(mdl_path):
            os.makedirs(mdl_path)
        for path in pose_valid[cls][mdl]:
            filename = path.split('/')[-1]
            shutil.move(path+'.png', os.path.join(mdl_path, filename+'.png'))
            shutil.move(path+'.npy', os.path.join(mdl_path, filename+'.npy'))
    
import IPython; IPython.embed()