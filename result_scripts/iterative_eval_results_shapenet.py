# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:24:38 2018

@author: bokorn
"""

import torch

from generic_pose.models.pose_networks import gen_pose_net
from generic_pose.eval.iterative_evaluation import evaluateIterQuat, evaluateIterQuatBatch
import generic_pose.utils.transformations as tf_trans
from quat_math import randomQuatNear

import numpy as np
import cv2
import glob
import os

def genModelDict(model_path_file):
    model_dict = {}
    with open(model_path_file, 'r') as f:    
        filenames = f.read().split()
    for path in filenames:
        model = path.split('/')[-2]
        model_dict[model] = path
    return model_dict

def getModelFiles(render_dict, model_dict):
    models = []
    for fn in render_dict.values():
        models.append(model_dict[fn.split('/')[-1]])
    return models
    
def getMaxCheckpoint(weights_folder):
    max_checkpoint = -float('inf')
    files = glob.glob(os.path.join(weights_folder,'checkpoint_*.pth'), recursive=True)
    for filename in files:
        cp = int(filename.split('.')[-2].split('_')[-1])
        max_checkpoint = max(max_checkpoint, cp)
    return max_checkpoint

model_data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/model_sets/all_shapenet.txt'
train_models = '/home/bokorn/src/generic_pose/generic_pose/training_sets/folder_sets/all_class_train_0.txt'
valid_model_models = '/home/bokorn/src/generic_pose/generic_pose/training_sets/folder_sets/all_class_model_valid_0.txt'
valid_class_models = '/home/bokorn/src/generic_pose/generic_pose/training_sets/folder_sets/all_class_class_valid_0.txt'


model_dict = genModelDict(model_data_file)
train_dict = genModelDict(train_models)
valid_model_dict = genModelDict(valid_model_models)
valid_class_dict = genModelDict(valid_class_models)

train_models = getModelFiles(train_dict, model_dict)
valid_models = getModelFiles(valid_model_dict, model_dict)
valid_class_models = getModelFiles(valid_class_dict, model_dict)

train_models = [fn.replace('ssd0', 'scratch') for fn in train_models]
valid_models = [fn.replace('ssd0', 'scratch') for fn in valid_models]
valid_class_models = [fn.replace('ssd0', 'scratch') for fn in valid_class_models]

results_folder = '/home/bokorn/results/result_imgs/iter_full_shapenet/' 

weight_paths = ['/home/bokorn/results/shapenet/alexnet_reg_shapenet/2018-05-08_21-33-09/weights',
                '/home/bokorn/results/shapenet/alexnet_reg_shapenet_45deg/2018-05-08_21-33-09/weights',
                '/home/bokorn/results/shapenet/alexnet_reg_shapenet_90deg/2018-05-08_21-33-08/weights',
                '/home/bokorn/results/shapenet/alex_reg_shapenet_maxest_45deg/2018-05-08_21-33-08/weights',
                ]

weight_filenames = []
experiment_name = []
for path in weight_paths:
    checkpoint = getMaxCheckpoint(path)
    weight_filenames.append(os.path.join(path, 'checkpoint_{}.pth'.format(checkpoint)))
    experiment_name.append(path.split('/')[-3])

#import IPython; IPython.embed()

num_train_models = None
num_valid_models = None
num_samples = 3

iter_quats = [tf_trans.random_quaternion() for _ in range(num_samples)]
#query_quats = [tf_trans.random_quaternion() for _ in range(num_samples)]
query_quats = [randomQuatNear(q, np.pi/4)[0] for q in iter_quats]

filename = os.path.join(results_folder,'eval_quats.npz')
np.savez(filename, iter_quats=iter_quats, query_quats=query_quats)

for name, weight_file in zip(experiment_name, weight_filenames):
    print(name)
    model = gen_pose_net('alexnet', 'basic', output_dim = 4)
    model.load_state_dict(torch.load(weight_file))
    model.eval()
    model.cuda()
                                  
    train_errs = []
    #train_imgs = []
    #train_tgts = []
    for model_file in train_models[:num_train_models]:
        print(model_file)
        #[errs, iter_imgs, target_imgs] = evaluateIterQuatBatch(model, model_file,
        #                                                       iter_quats,query_quats,
        #                                                       num_samples=num_samples)
        
        errs = evaluateIterQuatBatch(model, model_file,
                                     iter_quats,query_quats,
                                     num_samples=num_samples,
                                     return_images=False)
        train_errs.append(errs)
        #train_imgs.append(iter_imgs)
        #train_tgts.append(target_imgs)

    valid_errs = []
    #valid_imgs = []
    #valid_tgts = []
    for model_file in valid_models[:num_valid_models]:
        print(model_file)
        #[errs, iter_imgs, target_imgs] = evaluateIterQuatBatch(model, model_file,
        #                                                       iter_quats,query_quats, 
        #                                                       num_samples=num_samples)
        errs = evaluateIterQuatBatch(model, model_file,
                                     iter_quats,query_quats, 
                                     num_samples=num_samples,
                                     return_images=False)
        valid_errs.append(errs)
        #valid_imgs.append(iter_imgs)
        #valid_tgts.append(target_imgs)

    valid_class_errs = []
    #valid_imgs = []
    #valid_tgts = []
    for model_file in valid_class_models[:num_valid_models]:
        print(model_file)
        #[errs, iter_imgs, target_imgs] = evaluateIterQuatBatch(model, model_file,
        #                                                       iter_quats,query_quats, 
        #                                                       num_samples=num_samples)
        errs = evaluateIterQuatBatch(model, model_file,
                                     iter_quats,query_quats, 
                                     num_samples=num_samples,
                                     return_images=False)
        valid_class_errs.append(errs)
        #valid_imgs.append(iter_imgs)
        #valid_tgts.append(target_imgs)
        
    filename = os.path.join(results_folder, name+'_errs.npz')
    np.savez(filename, 
             train_errs=train_errs,
             valid_errs=valid_errs,
             valid_class_errs=valid_class_errs)
    #filename = os.path.join(results_folder, name+'_imgs.npz')
    #np.savez(filename, 
    #         train_imgs=train_imgs,
    #         train_tgts=train_tgts,
    #         valid_imgs=valid_imgs,
    #         valid_tgts=valid_tgts)
                      
import IPython; IPython.embed()