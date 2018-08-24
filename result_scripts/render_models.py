# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:57:00 2018

@author: bokorn
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_renderer.syscall_renderer import renderView

import numpy as np
import cv2
import os

#results_folder = '/home/bokorn/results/result_imgs/iter_full/' 
#
#train_models = np.load(os.path.join(results_folder, 'train_models.npy'))
#train_models = [fn.replace('ssd0', 'scratch') for fn in train_models]
#valid_models = np.load(os.path.join(results_folder, 'valid_models.npy'))
#valid_models = [fn.replace('ssd0', 'scratch') for fn in valid_models]
#results_folder = '/home/bokorn/results/result_imgs/iter_full_all_car/' 


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

train_model_names = []
for model_file in train_models:
    rendered_images = renderView(model_file, [np.array([0,0,0,1])],
                                 camera_dist = 2,
                                 standard_lighting = True)
    name = "_".join(model_file.split('/')[-3:-1])
    train_model_names.append(name)
    cv2.imwrite(os.path.join(results_folder, 'renders/train_' + name + '.png'), rendered_images[0])

plt.plot(np.split(np.arange(len(train_model_names)),1))
plt.legend(train_model_names)
plt.savefig(results_folder +'renders/train_names.png')
plt.gcf().clear()

valid_model_names = []
for model_file in valid_models:
    rendered_images = renderView(model_file, [np.array([0,0,0,1])],
                                 camera_dist = 2,
                                 standard_lighting = True)
    name = "_".join(model_file.split('/')[-3:-1])
    valid_model_names.append(name)
    cv2.imwrite(os.path.join(results_folder, 'renders/valid_' + name + '.png'), rendered_images[0])

plt.plot(np.split(np.arange(len(valid_model_names)),1))
plt.legend(valid_model_names)
plt.savefig(results_folder +'renders/valid_names.png')
plt.gcf().clear()


import IPython; IPython.embed()
