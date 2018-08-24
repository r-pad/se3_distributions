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

def getModelFiles(render_folder, model_dict):
    render_files = glob.glob(render_folder + '/**/*.png', recursive=True)
    models = set()
    for path in render_files:
        [model_class, model_name] = path.split('/')[-3:-1]
        models.add(model_dict[model_name])
    return list(models)

def getMaxCheckpoint(weights_folder):
    max_checkpoint = -float('inf')
    files = glob.glob(os.path.join(weights_folder,'checkpoint_*.pth'), recursive=True)
    for filename in files:
        cp = int(filename.split('.')[-2].split('_')[-1])
        max_checkpoint = max(max_checkpoint, cp)
    return max_checkpoint

#results_folder = '/home/bokorn/results/result_imgs/iter_sub45/'
results_folder = '/home/bokorn/results/result_imgs/iter_full/' 
#model_data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/model_sets/cars_100_train.txt'
#train_data_folder = '/scratch/bokorn/data/renders/car_42_renders/train' 
#valid_model_folder = '/scratch/bokorn/data/renders/car_42_renders/valid'


#model_dict = genModelDict(model_data_file)
#train_models = getModelFiles(train_data_folder, model_dict)
#valid_models = getModelFiles(valid_model_folder, model_dict)
#np.save(os.path.join(results_folder, 'train_models.npy'), train_models)
#np.save(os.path.join(results_folder, 'valid_models.npy'), valid_models)

train_models = np.load(os.path.join(results_folder, 'train_models.npy'))
valid_models = np.load(os.path.join(results_folder, 'valid_models.npy'))
results_folder = '/home/bokorn/results/result_imgs/iter_full_45/' 

dot_q_path = '/home/bokorn/results/alex_reg_car42_maxdot_45deg_quat/2018-04-11_06-22-46/weights'
#dot_a_path = '/home/bokorn/results/alex_reg_car42_maxdot_45deg_axis/2018-04-11_06-25-46/weights'
#mult_q_path = '/home/bokorn/results/alex_reg_car42_maxmult_45deg_quat/2018-04-11_06-28-02/weights'
#mult_a_path = '/home/bokorn/results/alex_reg_car42_maxmult_45deg_axis/2018-04-11_06-39-32/weights'
#true_q_path = '/home/bokorn/results/alex_reg_car42_maxtrue_45deg_quat/2018-04-11_06-52-10/weights'
#true_a_path = '/home/bokorn/results/alex_reg_car42_maxtrue_45deg_axis/2018-04-11_06-50-11/weights'

#quat_weight_paths = [dot_q_path,mult_q_path,true_q_path]
quat_weight_paths = [dot_q_path]#,true_q_path]
#aa_weight_paths = [dot_a_path,mult_a_path,true_a_path]

checkpoint_maxs = []
for path in quat_weight_paths:
    checkpoint_maxs.append(getMaxCheckpoint(path))
checkpoint = min(checkpoint_maxs)

weight_filenames = []
experiment_name = []
for path in quat_weight_paths:
    weight_filenames.append(os.path.join(path, 'checkpoint_{}.pth'.format(checkpoint)))
    experiment_name.append(path.split('/')[4])

experiment_name.append('alex_reg_car42')
weight_filenames.append('/home/bokorn/results/alex_reg_car42/2018-03-17_20-10-18/weights/best_quat.pth')

#experiment_name.append('alex_reg_car42_90deg')
#weight_filenames.append('/home/bokorn/results/alex_reg_car42_90deg/2018-03-30_23-23-54/weights/best_quat.pth')

experiment_name.append('alex_reg_car42_45deg')
weight_filenames.append('/home/bokorn/results/alex_reg_car42_45deg/2018-03-30_23-24-02/weights/best_quat.pth')
import IPython; IPython.embed()

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
    filename = os.path.join(results_folder, name+'_errs.npz')
    np.savez(filename, 
             train_errs=train_errs,
             valid_errs=valid_errs)
    #filename = os.path.join(results_folder, name+'_imgs.npz')
    #np.savez(filename, 
    #         train_imgs=train_imgs,
    #         train_tgts=train_tgts,
    #         valid_imgs=valid_imgs,
    #         valid_tgts=valid_tgts)
                      
import IPython; IPython.embed()